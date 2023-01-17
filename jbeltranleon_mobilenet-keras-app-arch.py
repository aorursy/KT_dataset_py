from keras.applications.mobilenet import MobileNet
model = MobileNet(input_shape=(224,224,3), include_top=True, weights='imagenet')
model.summary()
model.get_config()
model_no_top = MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')
model_no_top.summary()
model_no_top.get_config()