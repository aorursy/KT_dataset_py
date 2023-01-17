import os

from wand.image import Image as Img

files = next(os.walk('../input/cvpr2019/CVPR2019/papers'))[2] 

examplePaper = '../input/cvpr2019/CVPR2019/papers/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.pdf'

exampleAbstract = '../input/cvpr2019/CVPR2019/abstracts/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.txt' 
Img(filename=examplePaper, resolution=300)
print("Total number of papers in CVPR 2019 Papers dataset: ", len(files),'\n')

print("Abstract for first paper:",'\n\n','Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.txt','\n')

with open(exampleAbstract) as f: 

    print (f.read(1000))