import cv2
import json

import matplotlib.pyplot as plt
train_captions = json.load(open("../input/captions_train-val2014/annotations/captions_train2014.json"))
train_captions.keys()
print("Length: ", len(train_captions["images"]))
image_description = train_captions["images"][5]
image_description
image = cv2.imread(f"../input/train2014/train2014/{image_description['file_name']}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(30, 10))
plt.imshow(image)
plt.axis("off")
plt.show()
[
    caption["caption"]
    for caption in train_captions["annotations"]
    if caption["image_id"] == image_description["id"]
]