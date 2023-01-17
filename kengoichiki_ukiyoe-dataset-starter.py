!ls /kaggle/input
!ls /kaggle/input/the-metropolitan-museum-of-art-ukiyoe-dataset
from pathlib import Path



PATH = Path('/kaggle/input/the-metropolitan-museum-of-art-ukiyoe-dataset/images')

files = sorted([str(p) for p in PATH.glob('*/*.jpg')])

len(files), files[0]
labels = [p.split('/')[-2] for p in files]

len(labels), labels[0]
from collections import Counter



freq = Counter(labels).most_common()

freq
classes = [d[0] for d in freq]

len(classes)
cls_to_id = {c: i for i, c in enumerate(classes)}

cls_to_id
files_hiroshige = sorted([str(p) for p in PATH.glob('Utagawa_Hiroshige/*.jpg')])

len(files_hiroshige)
from PIL import Image
Image.open(files_hiroshige[0])