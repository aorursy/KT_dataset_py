from tqdm import tqdm

from time import sleep



text = ""

for char in tqdm(["a", "b", "c", "d"]):

    sleep(0.25)

    text = text + char
with tqdm(total=100) as pbar:

    for i in range(10):

        sleep(0.1)

        pbar.update(10)
from tqdm import tnrange, tqdm_notebook

from time import sleep



for i in trange(3, desc='1st loop'):

    for j in tqdm(range(100), desc='2nd loop'):

        sleep(0.01)