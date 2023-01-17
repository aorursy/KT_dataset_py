from pathlib import Path

import zipfile

img_root = Path('/kaggle/working/AODimgs/')
with zipfile.ZipFile('/kaggle/working/AODimgs.zip', 'w') as z:

    for img_name in img_root.iterdir():

        z.write(img_name)

    z.close()
ls /kaggle/working/AODimgs