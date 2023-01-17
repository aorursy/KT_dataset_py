import random
import itertools
import copy
import json

import numpy
import pylab
import torch

from torchvision import transforms
class StimuliDataset(torch.utils.data.Dataset):
    """
    A synthetic dataset with generated trajectories of different sized squares where
    the final generated videos could be labelled in a few simple trajectory categories
    like "middle-sized square moving to the right", "small square moving upwards", etc.

    The trajectories remain straight lines and the squares move along them linearly.

    For convenience we first create movies with double width and height
    and at the end we select a window.

    Original source: "Next-frame prediction with Conv-LSTM" by Keras authors.
    """
    def __init__(self, frames=15, width=40, height=40, channels=1,
                 squares_min=1, squares_max=2, size_min=1, size_max=3,
                 subsample=1.0, filename="", transform=None):
        super(StimuliDataset, self).__init__()

        self.frames = frames
        self.width = width
        self.height = height
        self.channels = channels

        self.square_number_range = (squares_min, squares_max)
        self.square_size_range = (size_min, size_max)

        if filename == "":
            square_states = []
            for s in range(*self.square_size_range):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for x0 in range(self.width):
                            for y0 in range(self.height):
                                square_states.append((s, dx, dy, x0, y0))

            self._idefs = []
            for squares in range(*self.square_number_range):
                self._idefs += [i for i in itertools.product(square_states, repeat=squares)]

            if subsample < 1.0:
                sample_count = int(subsample * len(self._idefs))
                self._idefs = random.sample(self._idefs, sample_count)
        else:
            self.load_from_file(filename)

        self.transform = transform

    def __len__(self):
        return len(self._idefs)

    def __getitem__(self, i):
        row, col = 2 * self.height, 2 * self.width
        x_range = int(self.width / 2), int(self.width + self.width / 2)
        y_range = int(self.height / 2), int(self.height + self.height / 2)
        frames = numpy.zeros((self.frames, row, col, self.channels), dtype=numpy.float)

        # Generate the frames of moving squares
        label = 0
        for j in range(len(self._idefs[i])):
            s, dx, dy, x0, y0 = self._idefs[i][j]
            # Calculate the label from the number of squares, square sizes, and directions
            label += (self.classes_per_square ** j) * (dx + 1 + 3 * (dy + 1) + 9 * (s - self.square_size_range[0]))

            for t in range(self.frames):
                xt = x_range[0] + x0 + dx * t
                yt = y_range[0] + y0 + dy * t
                frames[t, max(xt - s, 0): max(xt + s, 0), max(yt - s, 0): max(yt + s, 0), :] += 1

                # Improve robustness by adding noise
                if numpy.random.randint(0, 2):
                    # Generate random sign (-1, 0, +1) for the perturbation
                    noise_f = (-1)**numpy.random.randint(0, 2)
                    frames[t, max(xt - s - 1, 0): max(xt + s + 1, 0), max(yt - s - 1, 0): max(yt + s + 1, 0), 0] += noise_f * 0.1

        # Cut to the final movie window
        frames = frames[::, x_range[0]:x_range[1], y_range[0]:y_range[1], ::]
        frames[frames >= 1] = 1

        frames = torch.Tensor(frames)
        if self.transform:
            frames = self.transform(frames)

        return frames, label

    def __sub__(self, dataset):
        diff = copy.copy(self)
        diff._idefs = list(set(self._idefs) - set(dataset._idefs))
        return diff

    @property
    def classes_per_square(self):
        """Calculate the number of available classes per square."""
        return 3 * 3 * (self.square_size_range[1] - self.square_size_range[0])

    @property
    def classes(self):
        """Calculate the number of available classes in total."""
        return self.classes_per_square ** (self.square_number_range[1] - self.square_number_range[0])

    def load_from_file(self, filename):
        """
        Load the dataset from a file.

        :param str filename: file name to load from
        """
        with open(filename, "r") as f:
            self._idefs = json.load(f)
        # resture notion of tuples to be able to set-subtract
        for i in range(len(self._idefs)):
            self._idefs[i] = tuple(tuple(self._idefs[i][j]) for j in range(len(self._idefs[i])))
        squares_min, squares_max = len(min(self._idefs, key=lambda x: len(x))), len(max(self._idefs, key=lambda x: len(x)))+1
        size_min, size_max = min([i[0] for i in self._idefs])[0], max([i[0] for i in self._idefs])[0]+1

        self.square_number_range = (squares_min, squares_max)
        self.square_size_range = (size_min, size_max)

    def save_to_file(self, filename):
        """
        Save the dataset to a file.

        :param str filename: file name to load from
        """
        with open(filename, "w") as f:
            json.dump(self._idefs, f)

    def summary(self, train=False):
        """
        Print a summary of the loaded dataset.

        :param bool train: whether the dataset is a train set or test set
        """
        print()
        print(f"Total number of {'train' if train else 'test'} samples: {len(self)}")
        print(f"Total number of classes: {self.classes}")
        print(f"Video shape (frames, witdth, height, channels): {self.frames}x{self.width}x{self.height}x{self.channels}")
        print(f"Number of squares: from {self.square_number_range[0]} to {self.square_number_range[1]-1}")
        print(f"Size of squares: from {self.square_size_range[0]} to {self.square_size_range[1]-1}")
        print()
batch_size, test_percentage = 10, 0.3
frames, width, height, channels = 5, 20, 20, 1
squares_min, squares_max = 1, 2
size_min, size_max = 1, 2
tfs = transforms.Compose([
        # provide three identical channels for grayscale RGB
        transforms.Lambda(lambda x: torch.cat((x, x, x), -1)),
        # reshape into (T, C, H, W) for easier convolutions
        transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
])
train_dataset = StimuliDataset(frames, width, height, channels,
                               filename="/kaggle/input/cognitive-stimuli/stimuli_train18.json",
                               transform=tfs)
test_dataset = StimuliDataset(frames, width, height, channels,
                              filename="/kaggle/input/cognitive-stimuli/stimuli_test18.json",
                              transform=tfs)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset.summary(train=True)
test_dataset.summary(train=False)
%matplotlib inline
for x, y in test_loader:
    for n in range(x.shape[0]):
        for t in range(x.shape[1]):
            fig = pylab.figure(figsize=(5, 5))

            pylab.title(f'Input trajectory {y[n]}', fontsize=10)
            pylab.imshow(x[n, t, ::, ::, 0], cmap='gray', vmin=0, vmax=1)
            pylab.show()

            pylab.close(fig)
    break

total_dataset = StimuliDataset(frames, width, height, channels,
                               squares_min, squares_max, size_min, size_max)
test_dataset = StimuliDataset(frames, width, height, channels,
                              squares_min, squares_max, size_min, size_max,
                              test_percentage)
train_dataset = total_dataset - test_dataset
assert train_dataset.classes == test_dataset.classes

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset.save_to_file("stimuli_train.json")
test_dataset.save_to_file("stimuli_test.json")