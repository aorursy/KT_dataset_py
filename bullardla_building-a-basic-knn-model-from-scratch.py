import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # image processing
import time # benchmarking
import bisect
import matplotlib.pyplot as plt
from glob import glob
import math



training_images = glob('../input/cat-and-dog/training_set/training_set/**/*.jpg')
plt.figure(figsize=(16, 16))
for i in range(1, 10):
    training_image = np.random.choice(training_images)
    plt.subplot(3, 3, i)
    plt.imshow(cv2.imread(training_image)) 
    plt.axis('off')


#im = cv2.imread("../input/cat-and-dog/training_set/training_set/cats/cat.1.jpg")
print("Training Set:")
print("Cat images: " + str(len(glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'))))
print("Dog images: " + str(len(glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg'))))
print("Test Set:")
print("Cat images: " + str(len(glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'))))
print("Dog images: " + str(len(glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg'))))
def jpg_to_arr(filename, shape=(1080, 1080)):
    '''
    Convert a JPG to a NumPy array of size (1080, 1080, 3).
    The (x, y)th pixel in the original JPG will be located 
    at jpg_to_arr(inp)[x][y], and will have three dimensions
    for its red, blue, and green color values, respectively.
    That is, if the pixel at (x, y) had the color #A51822,
    then the resulting array `arr` would have:
    arr[x][y] == [165 / 255, 24 / 255, 34 / 255]
    
    :param string filename: The path to the filename containing a JPG image. Undefined behavior otherwise
    :param tuple(int, int) shape: The resolution of the resulting NumPy array.
    '''
    
    img = cv2.imread(filename)
    resized_img = cv2.resize(img, shape) / 255
    return resized_img.astype(float)

print("Value of pixel (15, 27): " + str(jpg_to_arr(glob('../input/cat-and-dog/training_set/training_set/**/*.jpg')[0])[15][27]))
def dist(x, y):
    '''
    Computes the euclidean distance between two 3-dimensional vectors
    x and y.
    '''
    assert x.shape == y.shape  # This is why the reshaping in jpg_to_arr() is necessary
    
    total_dist = 0
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                total_dist += (x[i][j][k] - y[i][j][k])**2
    return np.sqrt(total_dist)

def faster_dist(x, y):
    '''
    Same as the above, but much faster
    '''
    return np.linalg.norm(x - y)

            
cat_img1 = jpg_to_arr(glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg')[0])
cat_img2 = jpg_to_arr(glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg')[1])
dog_img1 = jpg_to_arr(glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')[0])

t1 = time.time()
print("Euclidean Distance between two cat images: " + str(dist(cat_img1, cat_img2)))
print("Euclidean Distance between a cat and a dog image: " + str(dist(cat_img1, dog_img1)))
print("Execution Time: " + str(time.time() - t1))
t2 = time.time()
print("Euclidean Distance between two cat images: " + str(faster_dist(cat_img1, cat_img2)))
print("Euclidean Distance between a cat and a dog image: " + str(faster_dist(cat_img1, dog_img1)))
print("Execution Time: " + str(time.time() - t2))
class KNN:
    def __init__(self, k, shape=(64, 64), debug=False):
        self.k = k
        self.points = []  # A list of tuples, where `first == vector` and `second == class_val`.
        self.shape = shape
        self.img_read_time = 0
        self.img_resize_time = 0
        self.debug = debug
        
        
    def distance(self, x, y):
        '''
        Computes the euclidean distance between two 3-dimensional vectors
        x and y.
        '''
        
        return np.linalg.norm(x - y)
    
    
    def jpg_to_arr(self, filename):
        '''
        Convert a JPG to a NumPy array of size (dim1, dim2, 3).
        The (x, y)th pixel in the original JPG will be located 
        at jpg_to_arr(inp)[x][y], and will have three dimensions
        for its red, blue, and green color values, respectively.
        That is, if the pixel at (x, y) had the color #A51822,
        then the resulting array `arr` would have:
        arr[x][y] == [165 / 255, 24 / 255, 34 / 255]

        :param string filename: The path to the filename containing a JPG image. Undefined behavior otherwise
        :param tuple(int, int) shape: The resolution of the resulting NumPy array.
        '''
    
        t1 = time.time()
        img = cv2.imread(filename)
        t2 = time.time()
        resized_img = cv2.resize(img, self.shape) / 255
        t3 = time.time()
        self.img_read_time += t2 - t1
        self.img_resize_time += t3 - t2
        return resized_img.astype(np.float16)
    
    
    def classify_point_from_vector(self, vector):
        '''
        Classifies `vector` using the stored list of known points, `self.points`
        
        :param vector: A vector matching the dimensions of the vectors the KNN model was trained on.
        '''
        t1 = time.time()
        distances = []
        for index, point in enumerate(self.points):
            point_vector = point[0]
            dist = self.distance(vector, point_vector)
            bisect.insort(distances, [dist, index])
        class_count = {}
        for i in range(self.k):
            class_val = self.points[distances[i][1]][1]
            class_count[class_val] = class_count.get(class_val, 0) + 1
        if self.debug:
            t2 = time.time()
            print("Time to classify point: " + str(t2 - t1))
        return max(class_count, key=class_count.get)
    
    
    def classify_point_from_filename(self, filename):
        '''
        Classifies `filename` using the stored list of known points, `self.points`
        
        :param filename: A filename pointing to a JPG image.
        '''
        vector = self.jpg_to_arr(filename)
        return self.classify_point_from_vector(vector)
    
    
    def classify_points_from_filenames(self, filename_lists, class_names):
        '''
        Evalutes the model on a set of test images, where `filename_lists` is a list of lists where
        the Nth list contains all of the example image filenames for the Nth class in `class_names`
        
        :param filename_lists: A list of lists, where each list contains examples for a specific class
        :param class_names: A list of the class names
        '''

        assert len(filename_lists) == len(class_names)
                
        conf_matrix = [[0, 0], [0, 0]]
        for i in range(len(filename_lists)):
            for filename in filename_lists[i]:
                prediction = self.classify_point_from_filename(filename)
                prediction_index = class_names.index(prediction)
                conf_matrix[i][prediction_index] += 1
        return conf_matrix
        
    def classify_points_from_vectors(self, vectors_lists, class_names):
        '''
        Evalutes the model on a set of test images, where `vectors_lists` is a list of lists where
        the Nth list contains all of the example image filenames for the Nth class in `class_names`
        
        :param vectors_lists: A list of lists, where each list contains examples for a specific class
        :param class_names: A list of the class names
        '''

        assert len(vectors_lists) == len(class_names)
                
        conf_matrix = [[0, 0], [0, 0]]
        for i in range(len(vectors_lists)):
            for vector in vectors_lists[i]:
                prediction = self.classify_point_from_vector(vector)
                prediction_index = class_names.index(prediction)
                conf_matrix[i][prediction_index] += 1
        return conf_matrix
    
    
    def add_point(self, vector, class_val):
        '''
        Adds the data point `vector` to self.points
        '''
        
        self.points.append((vector, class_val))


    def load_dataset(self, training_examples, class_names):
        '''
        Populates `self.points` using `training_examples` and `class_names` to populate
        the first and second values in each tuple in `self.points`, respectively. 
        
        That is, training_examples is a list of lists, where the Nth list is a list of filenames
        that serve as examples for the Nth class_name. Thus, the length of `training_examples` must
        be equal to the length of `class_names`.
        
        :param training_examples: A list of lists, where each list contains examples for a specific class
        :param class_names: A list of the class names
        '''
        
        assert len(training_examples) == len(class_names)
        for i in range(len(class_names)):
            for example in training_examples[i]:
                self.add_point(self.jpg_to_arr(example), class_names[i])
        if self.debug:
            print("Total Image Read Time: " + str(self.img_read_time))
            print("Total Image Resize Time: " + str(self.img_resize_time))
    
    def get_distance_to_nearest_neighbor(self, vector):
        '''
        Get the distance between a point `vector` and its nearest neighbor in `self.points`.
        '''
        min_dist = math.inf
        for point,  _ in self.points:
            min_dist = min(min_dist, self.distance(vector, point))
        return min_dist
    
    def get_distances_to_n_nearest_neighbors(self, vector, n):
        '''
        Get the distances between a point `vector` and its nth nearest neighbors in `self.points`.
        '''
        dists = [self.distance(vector, point) for point, _ in self.points]
        return sorted(dists)[:n]
    
    def get_sorted_distances_to_neighbors_with_class_data(self, vector, class_val):
        '''
        Returns a sorted list of tuples, where each tuple is a pair of (distance, bool),
        where bool is True if the corresponding point belongs to class `class_val` or False
        otherwise.
        '''
        return sorted([(self.distance(vector, point), class_val == point_class_val) for point, point_class_val in self.points])


model = KNN(5, debug=True)
model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
                  glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
                  ["cats", "dogs"])
print("The model predicts the picture of a cat belongs to the class " + str(model.classify_point_from_filename(glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg')[0])))
'''
model = KNN(5, shape=(128, 128))
model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
                  glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
                  ["cats", "dogs"])
conf_matrix = model.classify_points([glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'),
                  glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')],
                  ["cats", "dogs"])
print("Confusion Matrix: " + str(conf_matrix))
'''
class MetricConstructor:
    def __init__(self, conf_matrix, class_names):
        self.cm = np.array(conf_matrix)
        self.class_names = class_names
        self.total = conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]
        
    
    def accuracy(self):
        return (self.cm[0][0] + self.cm[1][1]) / self.total
        
    
    def precision(self, invert=False):
        if invert:
            return self.cm[1][1] / (self.cm[1][1] + self.cm[0][1])
        else:
            return self.cm[0][0] / (self.cm[0][0] + self.cm[1][0])
    
    
    def recall(self, invert=False):
        if invert:
            return self.cm[1][1] / (self.cm[1][1] + self.cm[1][0])
        else:
            return self.cm[0][0] / (self.cm[0][0] + self.cm[0][1])
    
    
    def f1(self, invert=False):
        return (2 * self.precision(invert=invert) * self.recall(invert=invert)) / (self.precision(invert=invert) + self.recall(invert=invert))
    
    
    def print_metrics(self):
        
        print("Accuracy: " + str(self.accuracy()))
        print("Precision for {}: {}".format(self.class_names[0], self.precision()))
        print("Precision for {}: {}".format(self.class_names[1], self.precision(invert=True)))
        print("Recall for {}: {}".format(self.class_names[0], self.recall()))
        print("Recall for {}: {}".format(self.class_names[1], self.recall(invert=True)))
        print("F1 Score for {}: {}".format(self.class_names[0], self.f1()))
        print("F1 Score for {}: {}".format(self.class_names[1], self.f1(invert=True)))
        
        return self.accuracy()
    
    
conf_matrix = [[772, 239], [681, 331]]
metrics = MetricConstructor(conf_matrix, ["cats", "dogs"])
metrics.print_metrics()

def train_and_test_knn(k_arr, shape_arr):
    '''
    Trains and prints the evaluation metrics of KNN models tried on all image resolutions
    provided by `shape_arr` for all k-values provided by `k_arr`. That is, this function is
    O(n*m), where len(k_arr) == n and len(shape_arr) == m.
    
    The model is trained on the cat-and-dog training set, and then evaluated using the cat-and-dog
    test set, for each combination of values in `k_arr` and `shape_arr`.
    
    :param k_arr: A list of k-values to test the model with.
    :param shape_arr: A list of shape tuples to use as image resolutions for training the model on.
    '''
    for k in k_arr:
        print("k == " + str(k))
        for shape in shape_arr:
            print("shape == " + str(shape))
            model = KNN(k, shape=shape)
            model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
                      glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
                      ["cats", "dogs"])
            conf_matrix = model.classify_points_from_filenames([glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'),
                      glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')],
                      ["cats", "dogs"])
            MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()

            
#train_and_test_knn(range(1, 5), [(128, 128)])
'''
k == 1
Accuracy: 0.5289174493326743
Precision for cats: 0.5216417910447761
Precision for dogs: 0.5431918008784773
Recall for cats: 0.6913946587537092
Recall for dogs: 0.366600790513834
F1 Score for cats: 0.5946405784772437
F1 Score for dogs: 0.43775811209439525
k == 2
Accuracy: 0.5289174493326743
Precision for cats: 0.5216417910447761
Precision for dogs: 0.5431918008784773
Recall for cats: 0.6913946587537092
Recall for dogs: 0.366600790513834
F1 Score for cats: 0.5946405784772437
F1 Score for dogs: 0.43775811209439525
k == 3
Accuracy: 0.5516559565002471
Precision for cats: 0.536723163841808
Precision for dogs: 0.586490939044481
Recall for cats: 0.751730959446093
Recall for dogs: 0.35177865612648224
F1 Score for cats: 0.6262875978574372
F1 Score for dogs: 0.43977764051883883
k == 4
Accuracy: 0.5437469105289174
Precision for cats: 0.5308555399719496
Precision for dogs: 0.5745393634840871
Recall for cats: 0.7487636003956478
Recall for dogs: 0.33893280632411066
F1 Score for cats: 0.621255642183012
F1 Score for dogs: 0.42635177128651336
'''
#train_and_test_knn([i * 10 for i in range(1, 5)], [(128, 128)])
'''
k == 10
Accuracy: 0.5719228868017795
Precision for cats: 0.548953409858204
Precision for dogs: 0.6346863468634686
Recall for cats: 0.8041543026706232
Recall for dogs: 0.33992094861660077
F1 Score for cats: 0.6524879614767255
F1 Score for dogs: 0.44272844272844275
k == 20
Accuracy: 0.569451309935739
Precision for cats: 0.5461741424802111
Precision for dogs: 0.6390532544378699
Recall for cats: 0.8189910979228486
Recall for dogs: 0.3201581027667984
F1 Score for cats: 0.6553225168183616
F1 Score for dogs: 0.42659644502962474
k == 30
Accuracy: 0.5580820563519525
Precision for cats: 0.5382103200522534
Precision for dogs: 0.6199186991869918
Recall for cats: 0.8150346191889218
Recall for dogs: 0.30138339920948615
F1 Score for cats: 0.6483084185680568
F1 Score for dogs: 0.40558510638297873
k == 40
Accuracy: 0.5659911023232822
Precision for cats: 0.5444221776887107
Precision for dogs: 0.6273764258555133
Recall for cats: 0.8061325420375866
Recall for dogs: 0.32608695652173914
F1 Score for cats: 0.6499202551834131
F1 Score for dogs: 0.4291287386215865
'''
#train_and_test_knn([i * 100 for i in range(1, 5)], [(128, 128)])
'''
k == 100
Accuracy: 0.5892239248640633
Precision for cats: 0.5645624103299857
Precision for dogs: 0.643879173290938
Recall for cats: 0.7784371909000989
Recall for dogs: 0.40019762845849804
F1 Score for cats: 0.6544698544698544
F1 Score for dogs: 0.49360146252285186
k == 200
Accuracy: 0.606030647553139
Precision for cats: 0.5874183006535948
Precision for dogs: 0.6345431789737171
Recall for cats: 0.7111770524233432
Recall for dogs: 0.5009881422924901
F1 Score for cats: 0.643400447427293
F1 Score for dogs: 0.5599116510215351
k == 300
Accuracy: 0.6109738012852199
Precision for cats: 0.6014492753623188
Precision for dogs: 0.6224156692056583
Recall for cats: 0.6567754698318496
Recall for dogs: 0.5652173913043478
F1 Score for cats: 0.6278959810874705
F1 Score for dogs: 0.5924391506991196
k == 400
Accuracy: 0.6050420168067226
Precision for cats: 0.6043307086614174
Precision for dogs: 0.605759682224429
Recall for cats: 0.6073194856577646
Recall for dogs: 0.6027667984189723
F1 Score for cats: 0.605821410952146
F1 Score for dogs: 0.6042595344229816
'''
#train_and_test_knn([5, 50, 250, 400, 1000], [(2 ** i, 2 ** i) for i in [4, 5, 6, 7, 8]])

k_5 = [.563, .551, .547, .545, .546]
k_50 = [.583, .572, .570, .582, .581]
k_250 = [.615, .608, .613, .616, .616]
k_400 = [.617, .603, .597, .605, .600]
k_1000 = [.576, .584, .576, .577, .578]
k_accuraries = [(5, k_5), (50, k_50), (250, k_250), (400, k_400), (1000, k_1000)]

resolutions = [16, 32, 64, 128, 256]

plt.figure(figsize=(16, 16))
plt.rcParams.update({'font.size': 22})
for k, accuracies in k_accuraries:
    plt.plot(resolutions, accuracies, label="k == " + str(k))

plt.title("Model Performance")
plt.xlabel("Image Resolution")
plt.ylabel("Test Set Accuracies")
plt.grid()
plt.legend(title="KNN k value")
plt.show()
#train_and_test_knn([5, 50, 250, 400, 1000], [(2 ** i, 2 ** i) for i in range(1, 4)])

k_5 = [.532, .557, .581, .563, .551, .547, .545, .546]
k_50 = [.533, .574, .603, .583, .572, .570, .582, .581]
k_250 = [.531, .580, .595, .615, .608, .613, .616, .616]
k_400 = [.519, .571, .597, .617, .603, .597, .605, .600]
k_1000 = [.519, .567, .571, .576, .584, .576, .577, .578]
k_accuraries = [(5, k_5), (50, k_50), (250, k_250), (400, k_400), (1000, k_1000)]

resolutions = [2, 4, 8, 16, 32, 64, 128, 256]

plt.figure(figsize=(16, 16))
plt.rcParams.update({'font.size': 22})
for k, accuracies in k_accuraries:
    plt.plot(resolutions, accuracies, label="k == " + str(k))

plt.title("Model Performance")
plt.xlabel("Image Resolution")
plt.ylabel("Test Set Accuracies")
plt.grid()
plt.legend(title="KNN k value")
plt.show()
img = glob('../input/cat-and-dog/training_set/training_set/**/*.jpg')[0]
plt.figure(figsize=(16, 16))
for i in range(1, 10):
    data = jpg_to_arr(img, shape=(2 ** i, 2 ** i))
    plt.subplot(3, 3, i)
    plt.title("{}x{}x3".format(2 ** i, 2 ** i))
    plt.imshow(data) 
    plt.axis('off')
class CatOrDog:
    '''
    A simple data class.
    tail_length - {0: no tail, 1: short tail, 2: long tail}
    fur_color - {0: brown, 1: grey, 2: white, 3: black, 4: orange}
    pupil_type - {0: round, 1: vertical-slit}
    whiskers - {0: false, 1: true}
    class_val - {0: dog, 1: cat}
    '''
    
    def __init__(self, weight, tail_length, fur_color, age, pupil_type, whiskers, litter_size, class_val):
        self.weight = weight
        self.tail_length = tail_length
        self.fur_color = fur_color
        self.age = age
        self.pupil_type = pupil_type
        self.whiskers = whiskers
        self.litter_size = litter_size
        self.class_val = class_val
    
    
    def get_vector(self):
        '''
        Get a vector of the x values corresponding to this object (every value except for class_val).
        '''
        tail_length_one_hot = [self.tail_length == i for i in range(3)]
        fur_color_one_hot = [self.fur_color == i for i in range(5)]
        return np.array([self.weight, self.age, self.pupil_type, self.whiskers, self.litter_size] + tail_length_one_hot + fur_color_one_hot)
    
    
    def get_class(self):
        return self.class_val
        

def generate_dataset(cats_n, dogs_n):
    '''
    Populates a dummy dataset of cats and dogs. Note that this dataset may create dogs or cats that aren't realistic.
    This method only exists for instructional purposes.
    
    :param cats_n: Number of cat examples to generate.
    :param dogs_n: Number of dog examples to generate.
    '''
    
    dataset = []
    for i in range(cats_n):
        dataset.append(CatOrDog(np.random.uniform(0, 0.25), 1, np.random.randint(0, 5), np.random.uniform(0, 1), 1, 1, np.random.randint(2, 12) / 11, 1))
    for i in range(dogs_n):
        dataset.append(CatOrDog(np.random.uniform(0, 1), np.random.randint(0, 3), np.random.randint(0, 5), np.random.uniform(0, 0.8), 0, 1, np.random.randint(5, 8) / 11, 0))
    return dataset
    

    
training_set = generate_dataset(1000, 1000)
test_set = [[animal.weight for animal in generate_dataset(200, 0)], [animal.weight for animal in generate_dataset(0, 200)]]

model = KNN(5)
for animal in training_set:
    model.add_point(animal.weight, animal.get_class())

conf_matrix = model.classify_points_from_vectors(test_set, [1, 0])
MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()
training_set = generate_dataset(1000, 1000)
test_set = [[np.array([animal.weight, animal.pupil_type]) for animal in generate_dataset(200, 0)], [np.array([animal.weight, animal.pupil_type]) for animal in generate_dataset(0, 200)]]

model = KNN(5)
for animal in training_set:
    model.add_point(np.array([animal.weight, animal.pupil_type]), animal.get_class())

conf_matrix = model.classify_points_from_vectors(test_set, [1, 0])
MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()
training_set = generate_dataset(1000, 1000)
test_set = [[animal.get_vector() for animal in generate_dataset(200, 0)], [animal.get_vector() for animal in generate_dataset(0, 200)]]

model = KNN(5)
for animal in training_set:
    model.add_point(animal.get_vector(), animal.get_class())

conf_matrix = model.classify_points_from_vectors(test_set, [1, 0])
MetricConstructor(model.classify_points_from_vectors(test_set, [1, 0]), ["cats", "dogs"]).print_metrics()
accuracies = []
avg_dists = []
noisy_dimensions = []

for i in range(2, 6):
    print("Adding {} noisy dimensions".format(10 ** i))
    training_set = generate_dataset(1000, 1000)
    test_set = [[np.append(animal.get_vector(), np.random.random_sample(10 ** i)) for animal in generate_dataset(200, 0)], [np.append(animal.get_vector(), np.random.random_sample(10 ** i)) for animal in generate_dataset(0, 200)]]

    model = KNN(5)
    for animal in training_set:
        model.add_point(np.append(animal.get_vector(), np.random.random_sample(10 ** i)), animal.get_class())

    conf_matrix = model.classify_points_from_vectors(test_set, [1, 0])
    accuracies.append(MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics())
    
    total_dist = 0
    for test_point in test_set[0] + test_set[1]:
        total_dist += model.get_distance_to_nearest_neighbor(test_point)
    print("Average distance for any testing point to the nearest neighbor: " + str(total_dist / (len(test_set[0]) + len(test_set[1]))))
    
    avg_dists.append(total_dist / (len(test_set[0]) + len(test_set[1])))
    noisy_dimensions.append(10 ** i)
    
plt.figure(figsize=(6, 6))   
plt.plot(noisy_dimensions, accuracies)
plt.title("KNN when k == 5")
plt.xlabel("Number of Noisy Dimensions")
plt.ylabel("Test Set Accuracy")
plt.grid()
plt.show()
plt.figure(figsize=(6, 6))
plt.plot(noisy_dimensions, avg_dists)
plt.title("KNN when k == 5")
plt.xlabel("Number of Noisy Dimensions")
plt.ylabel("Average Distance to Nearest Neighbor")
plt.grid()
plt.show()
'''
nearest_any_dist = []
nearest_same_dist = []
nearest_diff_dist = []
furthest_any_dist = []
furthest_same_dist = []
furthest_diff_dist = []
for i in range(1, 8):
    print("Shape: {}x{}x3".format(2 ** i, 2 ** i))
    model = KNN(5, shape=(2 ** i, 2 ** i))
    model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
              glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
              ["cats", "dogs"])
    training_set_size = len(glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg')) + len(glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg'))
    test_set_cats = glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg')
    test_set_dogs = glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')
    nearest_neighbors_avg = np.zeros(training_set_size)
    nearest_same_class_neighbors_avg = np.zeros(4000)
    nearest_diff_class_neighbors_avg = np.zeros(4000)
    for filename in test_set_cats:
        vector = model.jpg_to_arr(filename)
        nearest_neighbors_avg += np.array(model.get_distances_to_n_nearest_neighbors(vector, training_set_size))
        nearest_neighbors_pairs = model.get_sorted_distances_to_neighbors_with_class_data(vector, "cats")
        nearest_neighbors_same, nearest_neighbors_diff = [dist for dist, same_class in nearest_neighbors_pairs if same_class],  [dist for dist, same_class in nearest_neighbors_pairs if not same_class]
        nearest_same_class_neighbors_avg += np.array(nearest_neighbors_same[:4000])
        nearest_diff_class_neighbors_avg += np.array(nearest_neighbors_diff[:4000])
    for filename in test_set_dogs:
        vector = model.jpg_to_arr(filename)
        nearest_neighbors_avg += np.array(model.get_distances_to_n_nearest_neighbors(vector, training_set_size))
        nearest_neighbors_pairs = model.get_sorted_distances_to_neighbors_with_class_data(vector, "dogs")
        nearest_neighbors_same, nearest_neighbors_diff = [dist for dist, same_class in nearest_neighbors_pairs if same_class],  [dist for dist, same_class in nearest_neighbors_pairs if not same_class]
        nearest_same_class_neighbors_avg += np.array(nearest_neighbors_same[:4000])
        nearest_diff_class_neighbors_avg += np.array(nearest_neighbors_diff[:4000])
    nearest_neighbors_avg /= len(test_set_cats) + len(test_set_dogs)
    nearest_same_class_neighbors_avg /= len(test_set_cats) + len(test_set_dogs)
    nearest_diff_class_neighbors_avg /= len(test_set_cats) + len(test_set_dogs)
    nearest_any_dist.append(nearest_neighbors_avg[0])
    nearest_same_dist.append(nearest_same_class_neighbors_avg[0])
    nearest_diff_dist.append(nearest_diff_class_neighbors_avg[0])
    furthest_any_dist.append(nearest_neighbors_avg[-1])
    furthest_same_dist.append(nearest_same_class_neighbors_avg[-1])
    furthest_diff_dist.append(nearest_diff_class_neighbors_avg[-1])
    print("Average distance to nearest neighbor: " + str(nearest_neighbors_avg[0]))
    print("Average distance to {}th neighbor: {}".format(training_set_size, nearest_neighbors_avg[-1]))
    print("Average distance to nearest same-class neighbor: " + str(nearest_same_class_neighbors_avg[0]))
    print("Average distance to 4000th same-class neighbor: " + str(nearest_same_class_neighbors_avg[-1]))
    print("Average distance to nearest diff-class neighbor: " + str(nearest_diff_class_neighbors_avg[0]))
    print("Average distance to 4000th diff-class neighbor: " + str(nearest_diff_class_neighbors_avg[-1]))

plt.figure(figsize=(8, 8))   
dims = [2 ** i for i in range(1, 8)]
plt.plot(dims, nearest_any_dist, label="Nearest Neighbor (Any class)")
plt.plot(dims, nearest_same_dist, label="Nearest Neighbor (Same class)")
plt.plot(dims, nearest_diff_dist, label="Nearest Neighbor (Diff class)")
plt.plot(dims, furthest_any_dist, label="Furthest Neighbor (Any class)")
plt.plot(dims, furthest_same_dist, label="Furthest Neighbor (Same class)")
plt.plot(dims, furthest_diff_dist, label="Furthest Neighbor (Diff class)")
plt.title("Distance to Nearest Neighbors")
plt.xlabel("Width/Height of Resized Images (Pixels)")
plt.ylabel("Distance to Neighbor")
plt.legend()
plt.grid\
plt.show()
'''
class ImprovedKNN(KNN):
    def __init__(self, k, shape=(64, 64), debug=False, color_flag=cv2.IMREAD_COLOR):
        '''
        :param int color_flag: Passed as a flag to `iv2.imread()`, denoting the method in which images should be read.
        '''
        KNN.__init__(self, k, shape, debug)
        self.color_flag = color_flag
        
        
    def run_pca(self, min_explained_var):
        '''
        Performs PCA to calculate a lower-dimensional representation space for the associated data stored
        in `self.points`.
        
        :param min_explained_var: The minimum variance that has to be explained by the lower-dimensional 
        representation space. More plainly, the explained variance of the selected principal components
        will be at least `min_explained_var`.
        '''
        self.mean = np.mean([np.array(el[0]).flatten() for el in self.points], axis=0)
        covariance_matrix = np.cov((np.array([np.array(el[0]).flatten() for el in self.points]) - self.mean).T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        del covariance_matrix
        self.explained_variances = eigenvalues / np.sum(eigenvalues)
        self.eigenvectors = np.array([])
        explained_var = 0
        i = 0
        while explained_var < min_explained_var:
            explained_var += self.explained_variances[i]
            i += 1
        self.eigenvectors = eigenvectors[:i]
        print("{} components are needed to explain {} variance in the dataset".format(i, explained_var))
        
        
    def transform_point(self, vector):
        '''
        Transforms the vector to the lower-dimensional respresentational space calculated by the PCA. As such,
        `self.run_pca()` must be called before invoking this method.
        '''
        return np.dot(np.array(vector).flatten() - self.mean, self.eigenvectors.T)
    
    
    def transform_points(self):
        self.points = [(self.transform_point(point), class_val) for point, class_val in self.points]
            
    
    def classify_point_from_vector(self, vector):
        t1 = time.time()
        distances = []
        vector = self.transform_point(vector)
        for index, point in enumerate(self.points):
            point_vector = point[0]
            dist = self.distance(vector, point_vector)
            bisect.insort(distances, [dist, index])
        class_count = {}
        for i in range(self.k):
            class_val = self.points[distances[i][1]][1]
            class_count[class_val] = class_count.get(class_val, 0) + 1
        if self.debug:
            t2 = time.time()
            print("Time to classify point: " + str(t2 - t1))
        return max(class_count, key=class_count.get)
    
    def jpg_to_arr(self, filename):
        '''
        Convert a JPG to a NumPy array of size (dim1, dim2, 3).
        The (x, y)th pixel in the original JPG will be located 
        at jpg_to_arr(inp)[x][y], and will have three dimensions
        for its red, blue, and green color values, respectively.
        That is, if the pixel at (x, y) had the color #A51822,
        then the resulting array `arr` would have:
        arr[x][y] == [165 / 255, 24 / 255, 34 / 255]

        :param string filename: The path to the filename containing a JPG image. Undefined behavior otherwise
        :param tuple(int, int) shape: The resolution of the resulting NumPy array.
        '''
    
        t1 = time.time()
        img = cv2.imread(filename, self.color_flag)
        t2 = time.time()
        resized_img = cv2.resize(img, self.shape) / 255
        t3 = time.time()
        self.img_read_time += t2 - t1
        self.img_resize_time += t3 - t2
        return resized_img.astype(np.float16)
    
    
    def graph_data(self, class_color_map):
        '''
        Graph the data points in `self.points` by using PCA to reduce the dimensionality of the data to two.
        :param class_color_map: A dict mapping the names of classes to the color that points belonging to the
        classes should have on the scatter plot.
        '''
        mean = np.mean([np.array(el[0]).flatten() for el in self.points], axis=0)
        covariance_matrix = np.cov((np.array([np.array(el[0]).flatten() for el in self.points]) - mean).T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        del covariance_matrix
        explained_var = np.sum(eigenvalues[:2] / np.sum(eigenvalues))
        print("{} of the variance is explained".format(explained_var))
        points = [(np.dot(np.array(point).flatten() - mean, eigenvectors[:2].T), class_val) for point, class_val in self.points]
        for point, class_val in points:
            plt.scatter(point[0], point[1], color=class_color_map[class_val])
        plot_title = ", ".join([k + " == " + v for (k, v) in class_color_map.items()])
        plt.title(plot_title)
        plt.grid()
        plt.show()
        
    
def train_and_test_improved_knn(k_arr, shape_arr, explained_var_arr, color_flag_arr):
    '''
    Same as `train_and_test_knn()`, but for the ImprovedKNN. Takes two additional parameters.
    Like `train_and_test_knn()`, this method evaluates models for every
    combination of input parameters - as such, it has O(n * m * o * p) runtime, where the constants
    are the length of the input parameter arrays, respectively.
    
    :param k_arr: A list of k-values to test the model with.
    :param shape_arr: A list of shape tuples to use as image resolutions for training the model on.
    :param explained_var_arr: A list of minimum explained variances to use to run the PCA with. 
    :param color_flag_arr: A list of color_flags to construct the ImprovedKNN model with.
    '''
    for k in k_arr:
        print("k == " + str(k))
        for shape in shape_arr:
            print("shape == " + str(shape))
            for explained_var in explained_var_arr:
                print("min_explained_var == " + str(explained_var))
                for color_flag in color_flag_arr:
                    print("color_flag == " + str(color_flag))
                    model = ImprovedKNN(k, shape=shape, color_flag=color_flag)
                    model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
                              glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
                              ["cats", "dogs"])
                    model.run_pca(explained_var)
                    model.transform_points()
                    conf_matrix = model.classify_points_from_filenames([glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'),
                              glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')],
                              ["cats", "dogs"])
                    MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()

model = ImprovedKNN(k, shape=(64, 64))
model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
          glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
          ["cats", "dogs"])
model.graph_data({"cats": "red", "dogs": "blue"})

train_and_test_improved_knn([5, 25], [(64, 64)], [0.9, 0.75], [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class SKLearnKNN(KNN):
    def fit(self):
        self.knn = KNeighborsClassifier(self.k)
        self.knn.fit([np.array(el[0]).flatten() for el in self.points], [el[1] for el in self.points])
    
    def classify_point_from_vector(self, vector):
        t1 = time.time()
        prediction = self.knn.predict(vector.flatten().reshape(1, -1))
        if self.debug:
            t2 = time.time()
            print("Time to classify point: " + str(t2 - t1))
        return prediction
    
class SKLearnKNNwithPCA(KNN):
    def fit(self, n_components):
        self.pca = PCA(n_components)
        self.knn = KNeighborsClassifier(self.k)
        transformed_points = self.pca.fit_transform([np.array(el[0]).flatten() for el in self.points])
        self.points = [(point, el[1]) for point, el in zip(transformed_points, self.points)]
        self.knn.fit([np.array(el[0]).flatten() for el in self.points], [el[1] for el in self.points])
        
    def classify_point_from_vector(self, vector):
        t1 = time.time()
        transformed_vector = self.pca.transform(vector.flatten().reshape(1, -1))
        prediction = self.knn.predict(transformed_vector)
        if self.debug:
            t2 = time.time()
            print("Time to classify point: " + str(t2 - t1))
        return prediction
    
    def graph_data(self, class_color_map):
        pca = PCA(2)
        transformed_points = pca.fit_transform([np.array(el[0]).flatten() for el in self.points])
        for transformed_point, el in zip(transformed_points, self.points):
            plt.scatter(transformed_point[0], transformed_point[1], color=class_color_map[el[1]][1])
        plot_title = ", ".join([k + " == " + v for (k, v) in class_color_map])
        plt.title(plot_title)
        plt.grid()
        plt.show()
            
        

model = SKLearnKNN(5, shape=(64, 64))
model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
          glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
          [0, 1])
model.fit()
conf_matrix = model.classify_points_from_filenames([glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'),
                          glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')],
                          [0, 1])
MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()

model = SKLearnKNNwithPCA(5, shape=(64, 64))
model.load_dataset([glob('../input/cat-and-dog/training_set/training_set/cats/*.jpg'),
          glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')],
          [0, 1])
model.fit(300)
conf_matrix = model.classify_points_from_filenames([glob('../input/cat-and-dog/test_set/test_set/cats/*.jpg'),
                          glob('../input/cat-and-dog/test_set/test_set/dogs/*.jpg')],
                          [0, 1])
MetricConstructor(conf_matrix, ["cats", "dogs"]).print_metrics()
model.graph_data([("cats", "red"), ("dogs", "blue")])