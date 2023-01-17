import numpy as np

import pandas as pd

from torch.utils.data.sampler import BatchSampler, Sampler

from torch.utils.data import DataLoader, Dataset

from typing import Iterator, List, Optional, Union, Dict
class WeightedClassSampler(Sampler):

    """Abstraction over data sampler.



    Allows you to create stratified sample on unbalanced classes.

    """



    def __init__(

        self, labels: List[int], class_weights: Dict 

    ):

        """

        Args:

            labels (List[int]): list of class label

                for each elem in the dataset

            class_weights (dict): give dict of class

                how may time its repete

        """

        super().__init__(labels)



        labels = np.array(labels)

        samples_per_class = {

            label: int((labels == label).sum() * class_weights[label]) for label in set(labels)

        }

        





        self.lbl2idx = {

            label: np.arange(len(labels))[labels == label].tolist()

            for label in set(labels)

        }

        

        self.length = sum(samples_per_class.values())

        self.labels = labels

        self.samples_per_class = samples_per_class

        self.class_weights = class_weights





    def __iter__(self) -> Iterator[int]:

        """

        Yields:

            indices of stratified sample

        """

        indices = []

        for key in sorted(self.lbl2idx):

            replace_flag = self.class_weights[key] > 1

            indices += np.random.choice(

                self.lbl2idx[key], self.samples_per_class[key], replace=replace_flag

            ).tolist()

        assert len(indices) == self.length

        np.random.shuffle(indices)



        return iter(indices)





    def __len__(self) -> int:

        """

        Returns:

             length of result sample

        """

        return self.length
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
class BalanceDataset:

    def __init__(self, df):

        self.target = df.target.values

    

    def __len__(self):

        return len(self.target)

    

    def __getitem__(self, item):

        return self.target[item]

    

    def __get_labels__(self):

        return self.target
dataset = BalanceDataset(df=train)



data_loader = DataLoader(

    dataset,

    sampler= WeightedClassSampler(labels=dataset.__get_labels__(), class_weights= {0:1, 1:10}),

    batch_size=5

)



len(data_loader)
for i, d in enumerate(data_loader):

    print(d)

    if i == 10: break