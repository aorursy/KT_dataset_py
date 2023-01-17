import pydicom

from matplotlib import pyplot as plt

%matplotlib inline
bad_files = ['../input/osic-pulmonary-fibrosis-progression/train/ID00052637202186188008618/4.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/26.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/23.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/11.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/28.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/6.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/5.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/19.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/15.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/25.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/21.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/9.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/24.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/8.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/7.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/20.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/22.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/18.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/10.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/1.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/2.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/31.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/17.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/3.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/14.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/4.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/27.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/16.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/13.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/12.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/30.dcm',

'../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/29.dcm']



for i in bad_files:

    try:      

        pyd = pydicom.read_file(i)

        image_data = pyd.pixel_array

        plt.imshow(image_data, cmap=plt.cm.bone)

    except:

        print(i)
print(i)

pyd = pydicom.read_file(i)

image_data = pyd.pixel_array

plt.imshow(image_data, cmap=plt.cm.bone)