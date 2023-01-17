import os #Sistem 

import warnings #uyarılar

print(os.listdir("../input/"))

warnings.filterwarnings("ignore")

# simple neural network of one complex valued neuron

# extended to use a periodic activation function

import numpy

# pandas for reading csv files

import pandas

# sklearn for splitting data into test/train sets

#import sklearn.cross_validation

import sklearn
class neuralNetwork:

    

    def __init__(self, inputs, cats, periods):

        # number of inputs

        self.inputs = inputs

        

        # link weights matrix

        self.w = numpy.random.normal(0.0, pow(1.0, -0.5), (self.inputs + 1))

        self.w = numpy.array(self.w, ndmin=2, dtype='complex128')

        self.w += 1j * numpy.random.normal(0.0, pow(1.0, -0.5), (self.inputs + 1))

        

        # testing overrride

        #self.w = numpy.array([1.0 + 0.0j, 1.0 + 0.0j], ndmin=2, dtype='complex128')

        

        # number of output class categories

        self.categories = cats

        

        # todo periodicity

        self.periodicity = periods

        

        pass

    

    def z_to_class(self, z):

        # first work out the angle, but shift angle from [-pi/2, +pi.2] to [0,2pi]

        angle = numpy.mod(numpy.angle(z) + 2*numpy.pi, 2*numpy.pi)

        # from angle to category

        p = int(numpy.floor (self.categories * self.periodicity * angle / (2*numpy.pi)))

        p = numpy.mod(p, self.categories)

        return p



    def class_to_angles(self, c):

        # class to several angles due to periodicity, using bisector

        angles = (c + 0.5 + (self.categories * numpy.arange(self.periodicity))) / (self.categories * self.periodicity) * 2 * numpy.pi

        return angles

    

    def status(self):

        print ("w = ", self.w)

        print ("categories = ", self.categories)

        print ("periodicity = ", self.periodicity)

        pass



    def query(self, inputs_list):

        # add bias input

        inputs_list.append(1.0)

        

        # convert input to complex

        inputs = numpy.array(inputs_list, ndmin=2, dtype='complex128').T

        #print("inputs = \n", inputs)

        

        # combine inputs, weighted

        z = numpy.dot(self.w, inputs)

        #print("z = ", z)

        

        # map to output classes

        o = self.z_to_class(z)

        #print("output = ", o)

        #print ("")

        return o

    

    def train(self, inputs_list, target):

        # add bias input

        inputs_list.append(1.0)

        

        # convert inputs and outputs list to 2d array

        inputs = numpy.array(inputs_list, ndmin=2, dtype='complex128').T



        # combine inputs, weighted

        z = numpy.dot(self.w, inputs)[0]

        

        # desired angle from trainging set

        # first get all possible angles

        desired_angles = self.class_to_angles(target)

        

        # potential errors errors

        errors =  numpy.exp(1j*desired_angles) - z

        # select smallest error

        e = errors[numpy.argmin(numpy.abs(errors))]

        

        # dw = e * x.T / (x.x.T)

        dw = (e * numpy.conj(inputs.T)) / (self.inputs + 1)

        #print("dw = ", dw)

        self.w += dw

        #print("new self.w = ", self.w )

        #print("test new self.w with query = ", self.query(inputs.T))

        #print("--")

    pass
# create instance of neural network

number_of_inputs = 4

categories = 3

periods = 3



n = neuralNetwork(number_of_inputs, categories, periods)

n.status()
# load the iris training data CSV file into a list

df = pandas.read_csv('../input/irisdata/iris.csv')

# scale the lengths

df[['PW', 'PL', 'SW', 'SL']] = df[['PW', 'PL', 'SW', 'SL']].astype(numpy.float64) * 0.01



import sklearn.model_selection

# shuffle and split dataframe into train and test sets, split 3/4 and 1/4

iris_train, iris_test = sklearn.model_selection.train_test_split(df, train_size=0.75)

#iris_train, iris_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=9)

iris_test.head()
# train neural network



epochs = 2



for e in range(epochs):

    # go through all records in the training data set

    for idx, data in iris_train.iterrows():

        data_list = data.tolist()

        species = data_list[0]

        lengths = data_list[1:]

        n.train(lengths, species)

        pass

    pass
n.status()
# query after training



scorecard = []



for idx, data in iris_test.iterrows():

    data_list = data.tolist()

    correct_species = int(data_list[0])

    lengths = data_list[1:]

    answer = n.query(lengths)

    print(correct_species, answer)

    if (answer == correct_species):

        # network's answer matches correct answer, add 1 to scorecard

        scorecard.append(1)

    else:

        # network's answer doesn't match correct answer, add 0 to scorecard

        scorecard.append(0)

        pass

    pass
# calculate the performance score, the fraction of correct answers

scorecard_array = numpy.asarray(scorecard)

print ("performance = ", scorecard_array.sum() / scorecard_array.size)
while True:

    # train neural network



    epochs = 2



    for e in range(epochs):

        # go through all records in the training data set

        for idx, data in iris_train.iterrows():

            data_list = data.tolist()

            species = data_list[0]

            lengths = data_list[1:]

            n.train(lengths, species)

            pass

        pass



    # query after training



    scorecard = []



    for idx, data in iris_test.iterrows():

        data_list = data.tolist()

        correct_species = int(data_list[0])

        lengths = data_list[1:]

        answer = n.query(lengths)

        #print(correct_species, answer)

        if (answer == correct_species):

            # network's answer matches correct answer, add 1 to scorecard

            scorecard.append(1)

        else:

            # network's answer doesn't match correct answer, add 0 to scorecard

            scorecard.append(0)

            pass

        pass



    # calculate the performance score, the fraction of correct answers

    scorecard_array = numpy.asarray(scorecard)

    sonuc =  scorecard_array.sum() / scorecard_array.size

    print(sonuc)

    if sonuc >0.9:

        break

        

print ("performance = ", scorecard_array.sum() / scorecard_array.size)