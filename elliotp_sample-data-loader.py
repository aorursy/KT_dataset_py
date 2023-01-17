import os

import pandas as pd

from matplotlib import pyplot as plt

import random



class DataLoader():



    def __init__(self, base_dir):

        self.base_dir = base_dir

        self.front_files = os.listdir(os.path.join(base_dir, 'front/front'))

        self.side_files = os.listdir(os.path.join(base_dir, 'side/side'))

        self.labels = pd.read_csv(os.path.join(base_dir, 'labels_utf8.csv'), header=0, index_col=None, squeeze=True).to_dict()

        self.num_labeled_samples = len(self.labels['ID'])  # 69827



    def get_sample(self):

        i = random.randint(0, self.num_labeled_samples-1)

        label = {}

        for key in self.labels.keys():

            label[key] = self.labels[key][i]

        front_image = plt.imread(os.path.join(self.base_dir, 'front/front', label['ID']))

        side_image = plt.imread(os.path.join(self.base_dir, 'side/side', label['ID']))

        return front_image, side_image, label



if __name__ == '__main__':

    data_loader = DataLoader('/kaggle/input/idoc-mugshots/')

    front_sample, side_sample, label = data_loader.get_sample()

    plt.figure(1)

    plt.subplot(121)

    plt.imshow(front_sample)

    plt.axis('off')

    plt.subplot(122)

    plt.imshow(side_sample)

    plt.axis('off')

    print("The is prisoner number " + label['ID'] + ".")

    print("He" if label['Sex'] == "Male" else "She", "is " + label['Height'] + ' tall and weighs ' + label['Weight'] +

          ' has ' + label['Hair'].lower() + ' hair and ' + label['Eyes'].lower() + ' eyes and identifies racially as ' + label['Race'] + '.')

    if label['Sex Offender']:

        print("This is a sex offender.")

    print("Reason for imprisonment: ", label['Offense'])