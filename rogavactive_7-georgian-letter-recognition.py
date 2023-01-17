from os import listdir

import numpy as np

import pandas as pd

import cv2

import random

import scipy.ndimage

import scipy.misc

import pickle
class NN:



    def __init__(self, n_inputs):

        self.label_names = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ' , '-']

        # learning info

        self.n_iterations = 200

        self.l_rate = 0.1

        # layer info

        self.l_sizes = [n_inputs, 200 , 100 , 50, 10]

        self.n_layer = len(self.l_sizes)

        # generating biases and weights on every hidden layer

        self.biases = [np.random.randn(i, 1) for i in self.l_sizes[1:]]

        self.weights = [np.random.randn(j, i) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]



    # Activation function

    def sigmoid(self, s):

        return 1.0 / (np.exp(-s) + 1.0)



    # Derivative of activation function

    def sigmoid_der(self, s):

        return self.sigmoid(s) * (1.0 - self.sigmoid(s))



    # Forward propagation

    def forward(self, data):

        data = data.reshape(data.shape[0] , 1)

        curr = data

        for i in range(len(self.biases)):

            bias = self.biases[i]

            weight = self.weights[i]

            mult = np.dot(weight , curr)

            curr = self.sigmoid(mult + bias)

        

        return curr



    # Backward propagation

    def backward(self, X, y):

        X = X.reshape(X.shape[0] , 1)

        biases_err = [np.zeros((i, 1)) for i in self.l_sizes[1:]]

        weights_err = [np.zeros((j, i)) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]

        

        # forward propagation while saving a and z values

        a = [X]

        z = []

        for i in range(len(self.biases)):

            bias = self.biases[i]

            weight = self.weights[i]

            curr = a[-1]

            mult = np.dot(weight , curr)

            z.append(mult + bias)

            curr = self.sigmoid(mult + bias)

            a.append(curr)



        # backpropagation

        loss = (a[-1] - y) * self.sigmoid_der(z[-1])

        weights_err[-1] = np.dot(loss, a[-2].transpose())

        biases_err[-1] = loss

        

        for i in range(2 , self.n_layer):

            loss = np.dot(self.weights[-i + 1].transpose(), loss) * self.sigmoid_der(z[-i])

            weights_err[-i] = np.dot(loss, a[-i - 1].transpose())

            biases_err[-i] = loss



        #update weights and biases

        for i in range(len(self.biases)):

            self.weights[i] -= self.l_rate * weights_err[i]

            self.biases[i] -= self.l_rate * biases_err[i]



    def training(self, data):

        for i in range(self.n_iterations):

            print("iteration number " + str(i))

            random.shuffle(data)

            for j in range(len(data)):

                X = data[j][0]

                X = X.reshape(X.shape[0],1)

                y = data[j][1]

                y = y.reshape(y.shape[0],1)

                self.backward(X , y)

                

    def classify(self, data):

            ans = self.forward(data)

            res = [0] * len(ans)

            ind = -1

            for i in range(len(ans)):

                if ans[i] > 0.5:

                    res[i] = 1

                    ind = i

                else:

                    res[i] = 0

            if (sum(res) > 1):

                return '-'

            return self.label_names[ind]
class DataObject:

    # These are variables to prevent adding the same feature twice accidently.

    ROTATE = False

    SCALE = False

    BLUR = False

    NOISE = False



    def __init__(self, image):

        self.image_arr = image

        self.flat_arr_len = image.shape[0] * image.shape[1]



    def get_matrix(self):

        return self.image_arr



    def get_array(self, shape=None):

        shape = (self.flat_arr_len, 1) if shape is None else shape

        (thresh, im_bw) = cv2.threshold(self.image_arr.astype(np.uint8), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return ((np.reshape(im_bw,shape)).astype(float))/255.0



    def set_parent_features(self, parent_obj):

        self.ROTATE = parent_obj.ROTATE

        self.SCALE = parent_obj.SCALE

        self.BLUR = parent_obj.BLUR

        self.NOISE = parent_obj.NOISE
class DataFrame:



    data = {}

    test_data = {}

    letters = []

    DEFAULT_COLOR = 255.0



    # If images are white on black, pass false as a first argument, please.

    def __init__(self, black_on_white=True, max_from_class=700, root_dir="../input/georgianletters/letters/ასოები/",

                 height=25, width=25,test_perc=0.3):

        self.TEST = test_perc

        self.TRAIN = 1-test_perc

        self.HEIGHT = height

        self.WIDTH = width

        self.train = {}

        self.test = {}

        self.add_data(root_dir, max_from_class, black_on_white)



    # Taking parent (children of root_dir) folder names as labels, they should be only 1 letter long.

    # Data should be in labeled letter folders.

    # If images are white on black, pass false as a second argument, please.

    def add_data(self, root_dir, max_from_class, black_on_white=True):

        for letter in listdir(root_dir):

            if len(letter) > 1:

                continue

            count = 0

            images = listdir(root_dir + letter)

            random.shuffle(images)

            for image_name in images:

                count += 1

                if count > max_from_class:

                    continue

                img = cv2.imread(root_dir + letter + "/" + image_name, cv2.IMREAD_GRAYSCALE)

                if img is None:

                    print("wrong image path")

                else:

                    if not black_on_white:

                        img = 255 - img

                        self.DEFAULT_COLOR = 0.0

                    resized_img = cv2.resize(img, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_CUBIC)

                    if letter not in self.data:

                        self.data[letter] = []

                        self.letters.append(letter)

                    self.data[letter].append(DataObject(resized_img))



    # Rotate alphas are angles.

    def add_rotate_f(self, rotate_alphas=(-15, -5, 5, 15)):

        rotate_alphas = list(set(rotate_alphas))

        rotate_alphas = [i for i in rotate_alphas if i % 360 != 0]  # removes angles which are useless

        if len(rotate_alphas) == 0:

            return

        for letter in self.letters:

            appendix = []

            for sample in self.data[letter]:

                if not sample.ROTATE:

                    sample.ROTATE = True

                    for angle in rotate_alphas:

                        new_sample = scipy.ndimage.interpolation.rotate(sample.get_matrix(), angle,

                                                                        mode='constant',

                                                                        cval=self.DEFAULT_COLOR,

                                                                        reshape=False)

                        new_dataobject = DataObject(new_sample)

                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.

                        appendix.append(new_dataobject)

            self.data[letter].extend(appendix)



    # Scale alphas are pixels to add edges (then resize to original size).

    # Warning: alphas that are bigger than 3 or smaller than -3 . passing them would cause an error.

    def add_scale_f(self, scale_alphas=(2, 0)):

        scale_alphas = list(set([int(i) for i in scale_alphas]))

        if 0 in scale_alphas:

            scale_alphas.remove(0)

        if len(scale_alphas) == 0:

            return

        for alpha in scale_alphas:

            assert -4 <= alpha <= 4

            if not -4 <= alpha <= 4:

                print(str(alpha) + " is forbidden, please pass correct scale alphas")

                return

        for letter in self.letters:

            appendix = []

            for sample in self.data[letter]:

                if not sample.SCALE:

                    sample.SCALE = True

                    for pixels in scale_alphas:

                        if pixels > 0:

                            new_sample = np.c_[np.full((self.HEIGHT + 2 * pixels, pixels), self.DEFAULT_COLOR),

                                               np.r_[np.full((pixels, self.WIDTH), self.DEFAULT_COLOR),

                                                     sample.get_matrix(),

                                                     np.full((pixels, self.WIDTH), self.DEFAULT_COLOR)],

                                               np.full((self.HEIGHT + 2 * pixels, pixels), self.DEFAULT_COLOR)]

                        else:

                            pixels *= -1

                            new_sample = sample.get_matrix()[pixels:-pixels, pixels:-pixels]

                        new_sample = cv2.resize(new_sample, dsize=(self.WIDTH, self.HEIGHT),

                                                interpolation=cv2.INTER_CUBIC)

                        new_dataobject = DataObject(new_sample)

                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.

                        appendix.append(new_dataobject)

            self.data[letter].extend(appendix)



    # Sigmas are values for blur coefficient. How much pixels should be interpolated to neighbour pixels.

    # Please keep values between 0 < sigma < 1.

    def add_blur_f(self, sigmas=(.3, 0)):

        sigmas = list(set(sigmas))

        sigmas = [i for i in sigmas if 0 < i < 1]  # removes values which are forbidden

        if len(sigmas) == 0:

            return

        for letter in self.letters:

            appendix = []

            for sample in self.data[letter]:

                if not sample.BLUR:

                    sample.BLUR = True

                    for sigma in sigmas:

                        new_sample = scipy.ndimage.gaussian_filter(sample.get_matrix(), sigma=sigma)

                        new_dataobject = DataObject(new_sample)

                        new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.

                        appendix.append(new_dataobject)

            self.data[letter].extend(appendix)



    # noise is maximum value added or decreased(max.:100), dots are how many dots are changed.

    def add_noise_f(self, noise=20, dots=10):

        if dots < 1 or 0 < noise < 100:

            return

        for letter in self.letters:

            appendix = []

            for sample in self.data[letter]:

                if not sample.NOISE:

                    sample.NOISE = True

                    new_sample = np.copy(sample.get_matrix())

                    for _ in range(dots):

                        x = random.randint(0, self.WIDTH - 1)

                        y = random.randint(0, self.HEIGHT - 1)

                        if new_sample[y][x] > 200:

                            noise *= -1

                        elif new_sample[y][x] > 50:

                            noise = random.randint(-noise, noise)

                        new_sample[y][x] = new_sample[y][x] + noise

                    new_dataobject = DataObject(new_sample)

                    new_dataobject.set_parent_features(sample)  # To prevent accidently using same feature twice.

                    appendix.append(new_dataobject)

            self.data[letter].extend(appendix)



    def get_random(self, letter):

        return random.choice(self.data[letter])



    def get_train(self, letter):

        return self.train[letter]

    

    def get_test(self,letter):

        return self.test[letter]



    def get_letters(self):

        return self.letters

    

    def shuffle_train(self):

        for letter in self.letters:

            random.shuffle(self.train[letter])

    

    def divide_data(self):

        for letter in self.letters:

            list_len = len(self.data[letter])

            self.train[letter] = self.data[letter][:int(list_len*self.TRAIN)]

            self.test[letter] = self.data[letter][int(list_len*self.TRAIN):]



    def describe(self):

        print("data contains " + str(len(self.letters)) + " letters, ")

        total = 0

        for letter in self.letters:

            amount = len(self.data[letter])

            total += amount

            print(str(amount) + " - " + letter + "'s.")

        print("\nTOTAL: " + str(total) + " letters.")
allData = DataFrame()



# allData.add_rotate_f()



# allData.add_blur_f()



# allData.add_scale_f()



allData.divide_data()



allData.describe()
#Necessary labels mapped on np array, necesasry for classifiyng during learning process.

labels = {'უ' : np.array([1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]), 

          'ყ' : np.array([0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]), 

          'მ' : np.array([0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0]),

          'შ' : np.array([0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0]),

          'ძ' : np.array([0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0]),

          'წ' : np.array([0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0]),

          'ს' : np.array([0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0]),

          'ხ' : np.array([0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0]),

          'ლ' : np.array([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0]),

          'ჩ' : np.array([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1]) }
n_inputs = 625

net = NN(625)



train = []

for letter in allData.get_letters():

    for dataObj in allData.get_train(letter):

        numpy_arr = dataObj.get_array()

        train.append((numpy_arr , labels[letter]))
net.training(train)
data_list = []

for letter in allData.get_letters():

    for dataObj in allData.get_test(letter):

        numpy_arr = dataObj.get_array()

        data_list.append((numpy_arr, letter+"_test"))
classified_data = []

gamoicno = 0

ver_gamoicno = 0

for data, img_name in data_list:

    classified_label = net.classify(data)

    classified_data.append((img_name,classified_label))

    if img_name[0] == classified_label:

        gamoicno +=1

    else:

        ver_gamoicno +=1

print("gamoicno: " + str(gamoicno) + " ver gamoicno: " + str(ver_gamoicno))
stats = {}

letters = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ']

for letter in letters:

    stats[letter] = [0,0]

for img_name, label in classified_data:

    letter = img_name[0]

    stats[letter][1] +=1

    if letter == label:

        stats[letter][0] +=1

        

for letter in stats:

    coeff = stats[letter][0]/float(stats[letter][1])

    print(letter + " ამოიცნო " + str(coeff) + " სიზუსტით.")
stats = {}

letters = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ' ]

for letter in letters:

    stats[letter] = {}

for img_name, label in classified_data:

    letter = img_name[0]

    if letter != label:

        if label not in stats[letter]:

            stats[letter][label] = 0

        stats[letter][label] +=1

        

for letter in stats:

    for label in stats[letter]:

        print("ასოზე " + letter + " - " + label + " დააფრედიქთა " + str(stats[letter][label]) + " -ჯერ" )
filename = "./7_model.sav"

with open(filename , 'wb') as file:

    net_info = {

                "biases" : net.biases,

                "weights" : net.weights,

                }

    pickle.dump(net_info, file, 2 )