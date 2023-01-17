!pip3 install detecto
import matplotlib.pyplot as plt
from detecto.utils import read_image
from torchvision import transforms
from detecto.utils import normalize_transform
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto.core import DataLoader, Model
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # Note: all images with a size smaller than 800 will be scaled up in size
    transforms.Resize(800),
    transforms.ToTensor(),  # required
    normalize_transform(),  # required
])
dataset = Dataset('../input/annotations/annot.csv', '../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images', transform=custom_transforms)
image, targets = dataset[0]
show_labeled_image(image, targets['boxes'], targets['labels'])
# Specify all unique labels you're trying to predict
your_labels = ['face_with_mask', 'mask_colorful', 'face_no_mask',
       'face_with_mask_incorrect', 'mask_surgical', 'face_other_covering',
       'scarf_bandana', 'eyeglasses', 'helmet', 'face_shield',
       'sunglasses', 'hood', 'hat', 'goggles', 'hair_net', 'hijab_niqab',
       'other', 'gas_mask', 'balaclava_ski_mask', 'turban']
model = Model(your_labels)

losses = model.fit(dataset, epochs=1, verbose=True)

plt.plot(losses)
plt.show()
# model.save('detecto-v1.pth')
# model = Model.load('../input/detecto-model/detecto-v1.pth', your_labels)
image = read_image('../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/1799.jpg')
plt.imshow(image)
predictions = model.predict_top(image)
