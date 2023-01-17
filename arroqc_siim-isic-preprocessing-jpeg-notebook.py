import zipfile
from PIL import Image
from pathlib import Path
import io
img_list = Path('/kaggle/input/siim-isic-melanoma-classification/jpeg').glob('**/*.jpg')
with zipfile.ZipFile('train.zip', 'w') as trainzip:
    for i, img_fn in enumerate(img_list):
        small_img = Image.open(img_fn).resize((224, 224))
        image_file = io.BytesIO()
        small_img.save(image_file, 'PNG')
        zipped_filename = img_fn.parts[-2] + '/' + img_fn.stem + '.png'
        trainzip.writestr(zipped_filename, image_file.getvalue())
