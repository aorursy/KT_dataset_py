import os

import re

import tensorflow as tf

base_dir = "/kaggle/input"  

baseline_path = os.path.join(base_dir, "delg-saved-models/local_and_global/variables/variables")



RESNET = '50'

# RESNET = '101'

if RESNET == '50':

    backbone = tf.keras.applications.ResNet50(

        include_top=False, weights=None, input_tensor=None, input_shape=None,

        pooling=None

    )

elif RESNET == "101":

    backbone = tf.keras.applications.ResNet101(

        include_top=False, weights=None, input_tensor=None, input_shape=None,

        pooling=None

    )
# loading host baseline model weight names

def make_ckpt_dict(ckpt_path):

    ck_list = tf.train.list_variables(ckpt_path)

    ck_dict = {var[0]: var[1] for var in ck_list}

    return ckpt_path, ck_list, ck_dict



baseline_ckpt = make_ckpt_dict(baseline_path)



ckpt_file, ck_list, ck_dict = baseline_ckpt

assert len([key for key in ck_dict.keys() if key.find(f'resnet_v1_{RESNET}') > -1]) > 0   

# loading keras resnet variable names

keras_vars = {}

for var in backbone.variables:

    keras_vars[var.name] = var.shape.as_list()

# keras_vars


def keras_to_slim(keras_name="conv2_block3_1_bn/beta:0", resnet="101"):

    # convert keras resnet variable names into correspoinding slim variable names

    layer_dict = {'bn': 'BatchNorm/', 'conv': ''}

    var_dict = {'beta:0': 'beta', 'gamma:0': 'gamma', 'moving_mean:0': 'moving_mean', 'moving_variance:0': 'moving_variance', 'kernel:0':'weights'}

    keras_split = keras_name.replace("moving_", "moving-").split("_")

    if len(keras_split) > 2:

        conv_id, block_id, layer_id, layer_name = keras_split

        conv_id, block_id = int(re.sub(r"\D", "", conv_id)), int(re.sub(r"\D", "", block_id))

        layer_id = int(layer_id)

        layer_name, var_type = layer_name.split("/")

        var_type = var_type.replace("moving-", "moving_")

        assert keras_name == f"conv{conv_id}_block{block_id}_{layer_id}_{layer_name}/{var_type}"

    else:

        conv_id, layer_name = keras_split

        conv_id = int(re.sub(r"\D", "", conv_id))

        layer_name, var_type = layer_name.split("/")

        var_type = var_type.replace("moving-", "moving_")

        assert keras_name == f"conv{conv_id}_{layer_name}/{var_type}"

        assert conv_id == 1



    if conv_id > 1:

        if layer_id > 0:

            slim_name = f"resnet_v1_{resnet}/block{conv_id-1}/unit_{block_id}/bottleneck_v1/conv{layer_id}/{layer_dict[layer_name]}{var_dict[var_type]}"

        else:

            assert block_id == 1

            slim_name = f"resnet_v1_{resnet}/block{conv_id-1}/unit_{block_id}/bottleneck_v1/shortcut/{layer_dict[layer_name]}{var_dict[var_type]}"

    else:

        slim_name = f"resnet_v1_{resnet}/conv1/{layer_dict[layer_name]}{var_dict[var_type]}"

    assert slim_name in [var[0] for var in ck_list]



    return slim_name



print("assigning each tf slim variable to tf keras variable by variable name")

for i, keras_var in enumerate(backbone.variables):

    keras_name = keras_var.name

    print(f"{i}: {keras_name} ->")

    if keras_name.find("bias:0") > -1:

        keras_var.assign(tf.zeros_like(keras_var))

        print(f"\t\t bias = zeros:")

        continue



    slim_name = keras_to_slim(keras_name, resnet=RESNET)

    slim_var = tf.train.load_variable(ckpt_file, slim_name)

    assert keras_var.numpy().shape == slim_var.shape

    assert keras_var.numpy().dtype == slim_var.dtype



    keras_var.assign(slim_var)

    print(f"\t\t {slim_name}")

    ck_dict.pop(slim_name)
class GeM(tf.keras.layers.Layer):

    # from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py

    def __init__(self, p=3, epsilon=1e-6, **kwargs):

        super().__init__(**kwargs)

        self.init_p = p

        self.epsilon = epsilon



    def build(self, input_shape):



        if isinstance(input_shape, list) or len(input_shape) != 4:

            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')



        self.build_shape = input_shape

        self.p = self.add_weight(

            name='gem_power',

            shape=[1,],

            initializer=tf.keras.initializers.Constant(value=self.init_p),

            regularizer=None,

            trainable=self.trainable,

            dtype=tf.float32

            )

        super().build(input_shape)



    def call(self, x):

        x = tf.pow(tf.clip_by_value(x, self.epsilon, tf.reduce_max(x)), self.p)

        x = tf.keras.layers.GlobalAvgPool2D()(x)

        x = tf.pow(x, 1.0/self.p)

        return x



    def compute_output_shape(self, input_shape):

        return tf.TensorShape(input_shape.as_list()[:1] + input_shape.as_list()[:-1])



    def get_config(self):

        base_config = super().get_config()

        return {**base_config, "init_p": self.init_p, "epsilon": self.epsilon}



class DELGEmbed(tf.keras.Model):

    def __init__(self, embed_dim, backbone, **kwargs):

        super().__init__(**kwargs)

        self.backbone = backbone

        self.gem = GeM(p=3, name="gem", trainable=False)

        self.embed = tf.keras.layers.Dense(embed_dim, activation=None, bias_initializer='zeros', name="embed")



    def call(self, inputs, training=False):

        x = self.backbone(inputs, training=training)

        x = self.gem(x)

        x = self.embed(x)

        return x





delg_embed = DELGEmbed(2048, backbone, name="delg_global")

delg_embed.build((None, 224, 224, 3))



print("assigning remaining weights") 

embed_w = tf.train.load_variable(ckpt_file, "embed/weights")

embed_b = tf.train.load_variable(ckpt_file, "embed/biases")

delg_embed.layers[-1].variables[0].assign(embed_w)  

delg_embed.layers[-1].variables[1].assign(embed_b)  



ck_dict.pop("embed/weights")

ck_dict.pop("embed/biases")



assert len([key for key in ck_dict.keys() if key.find(f"resnet_v1_{RESNET}/") > -1]) == 0

assert len([key for key in ck_dict.keys() if key.find("embed/") > -1]) == 0
print("unsued weights in the baseline model")

for key, value in ck_dict.items():

    print(f"{key}: {value}")

delg_embed.summary()

ckpt_converted_dir = "./"

delg_embed.save_weights(os.path.join(ckpt_converted_dir, f"baseline_delg_global_res{RESNET}.h5"))

delg_embed.layers[0].save_weights(os.path.join(ckpt_converted_dir, f"baseline_backbone_res{RESNET}.h5"))

print("saving conveted weights for both global descriptor and backbone(resnet)")

os.listdir("./")