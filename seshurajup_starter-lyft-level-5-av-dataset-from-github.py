!pip install lyft-dataset-sdk
# Load the SDK

%matplotlib inline

from lyft_dataset_sdk.lyftdataset import LyftDataset



# Load the dataset

# Adjust the dataroot parameter below to point to your local dataset path.

# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps

!ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar

level5data = LyftDataset(data_path='.', json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)

# Not tested this module yet
level5data.list_scenes()
my_scene = level5data.scene[0]

my_scene
my_sample_token = my_scene["first_sample_token"]

# my_sample_token = level5data.get("sample", my_sample_token)["next"]  # proceed to next sample



level5data.render_sample(my_sample_token)
my_sample = level5data.get('sample', my_sample_token)

my_sample
level5data.list_sample(my_sample['token'])
level5data.render_pointcloud_in_image(sample_token = my_sample["token"],

                                      dot_size = 1,

                                      camera_channel = 'CAM_FRONT')
my_sample['data']
sensor_channel = 'CAM_FRONT'  # also try this e.g. with 'LIDAR_TOP'

my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])

my_sample_data
level5data.render_sample_data(my_sample_data['token'])
my_annotation_token = my_sample['anns'][16]

my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)

my_annotation
level5data.render_annotation(my_annotation_token)
my_instance = level5data.instance[100]

my_instance
instance_token = my_instance['token']

level5data.render_instance(instance_token)
print("First annotated sample of this instance:")

level5data.render_annotation(my_instance['first_annotation_token'])
print("Last annotated sample of this instance")

level5data.render_annotation(my_instance['last_annotation_token'])
level5data.list_categories()
level5data.category[2]
level5data.list_attributes()
for my_instance in level5data.instance:

    first_token = my_instance['first_annotation_token']

    last_token = my_instance['last_annotation_token']

    nbr_samples = my_instance['nbr_annotations']

    current_token = first_token



    i = 0

    found_change = False

    while current_token != last_token:

        current_ann = level5data.get('sample_annotation', current_token)

        current_attr = level5data.get('attribute', current_ann['attribute_tokens'][0])['name']



        if i == 0:

            pass

        elif current_attr != last_attr:

            print("Changed from `{}` to `{}` at timestamp {} out of {} annotated timestamps".format(last_attr, current_attr, i, nbr_samples))

            found_change = True



        next_token = current_ann['next']

        current_token = next_token

        last_attr = current_attr

        i += 1
level5data.sensor
level5data.sample_data[10]
level5data.calibrated_sensor[0]
level5data.ego_pose[0]
print("Number of `logs` in our loaded database: {}".format(len(level5data.log)))
level5data.log[0]
print("There are {} maps masks in the loaded dataset".format(len(level5data.map)))

#level5data.map[0]
#sensor_channel = 'LIDAR_TOP'

#my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])

# The following call can be slow and requires a lot of memory

#level5data.render_sample_data(my_sample_data['token'], underlay_map = True)
level5data.category[0]
cat_token = level5data.category[0]['token']

cat_token
level5data.get('category', cat_token)
level5data.sample_annotation[0]
one_instance = level5data.get('instance', level5data.sample_annotation[0]['instance_token'])

one_instance
ann_tokens = level5data.field2token('sample_annotation', 'instance_token', one_instance['token'])
ann_tokens_field2token = set(ann_tokens)



ann_tokens_field2token
ann_record = level5data.get('sample_annotation', one_instance['first_annotation_token'])

ann_record
ann_tokens_traverse = set()

ann_tokens_traverse.add(ann_record['token'])

while not ann_record['next'] == "":

    ann_record = level5data.get('sample_annotation', ann_record['next'])

    ann_tokens_traverse.add(ann_record['token'])
print(ann_tokens_traverse == ann_tokens_field2token)
catname = level5data.sample_annotation[0]['category_name']
ann_rec = level5data.sample_annotation[0]

inst_rec = level5data.get('instance', ann_rec['instance_token'])

cat_rec = level5data.get('category', inst_rec['category_token'])



print(catname == cat_rec['name'])
# Shortcut

channel = level5data.sample_data[0]['channel']



# No shortcut

sd_rec = level5data.sample_data[0]

cs_record = level5data.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

sensor_record = level5data.get('sensor', cs_record['sensor_token'])



print(channel == sensor_record['channel'])
level5data.list_categories()
level5data.list_attributes()
level5data.list_scenes()
my_sample = level5data.sample[10]

level5data.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
my_sample = level5data.sample[20]



# The rendering command below is commented out because it tends to crash in notebooks

# level5data.render_sample(my_sample['token'])
level5data.render_sample_data(my_sample['data']['CAM_FRONT'])
level5data.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5)
level5data.render_annotation(my_sample['anns'][22])
#my_scene_token = level5data.scene[0]["token"]

#level5data.render_scene_channel(my_scene_token, 'CAM_FRONT')
#level5data.render_scene(my_scene_token)
#level5data.render_egoposes_on_map(log_location='Palo Alto')
# put your code here

# hint: 

# next_sample_data = level5data.get('sample_data', my_sample_data["next"])

# gives you the next sample data entry