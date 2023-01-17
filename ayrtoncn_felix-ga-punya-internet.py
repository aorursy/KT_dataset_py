#Class Definition
import csv
from sklearn.impute import SimpleImputer

class InputDataInterpreter():
    def __init__(self, filename = ""):
        self.filename = filename
        self.data = []
        self.target = []
        self.processInputFile()

    def processInputFile(self):
        input_data = self.getInputFileContent()
        self.makeDatasetList(input_data)
        
        for i in range(len(self.data)):
            self.target[i] = int(self.target[i])
            for j in range(len(self.data[0])):
                self.data[i][j] = float(self.data[i][j])	

    def getInputFileContent(self):
        data_content = []
        
        with open(self.filename, newline='') as csvfile:
            file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
            
            for row in file_content:
                content_row = []
                for data in row:
                    content_row.append(data)
                data_content.append(content_row)

        return data_content[1:]

    def makeDatasetList(self, input_data):
        for row in input_data:
            self.target.append(row[0].split(',')[-1])
            self.data.append(row[0].split(',')[0:13])
        
        # there's an empty data
        self.data[548][6] = '?'

        self.patchUnknownData()

    def patchUnknownData(self):
        column_patch_method = ["median", "modus", "modus", "mean", \
        "mean", "modus", "modus", "mean", \
        "modus", "mean", "modus", "modus", "modus"]
        
        column_patch_values = self.getColumnPatchVal(column_patch_method)

        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == '?':
                    self.data[i][j] = column_patch_values[j]


    def getColumnPatchVal(self, patch_method):
        patch_val = []

        for i in range(len(patch_method)):
            if patch_method[i] == 'modus':
                patch_val.append(self.getDataModus(i))
            elif patch_method[i] == 'median':
                patch_val.append(self.getDataMedian(i))
            elif patch_method[i] == 'mean':
                patch_val.append(self.getDataMean(i))

        return patch_val

    def getDataModus(self, j):
        data_dict = {}

        for i in range(len(self.data)):
            if self.data[i][j] == '?':
                continue
            if str(self.data[i][j]) in data_dict:
                data_dict[str(self.data[i][j])] += 1
            else :
                data_dict[str(self.data[i][j])] = 0

        max_key = ''
        max_val = -1
        for key, val in data_dict.items():
            if val > max_val:
                max_key = key
                max_val = val

        return str(max_val)

    def getDataMedian(self, j):
        column_list = []

        for i in range(len(self.data)):
            if self.data[i][j] == '?':
                continue
            column_list.append(self.data[i][j])

        column_list.sort()
        median_idx = (len(column_list)//2) + 1

        return str(column_list[median_idx])

    def __is_int__(self, input):
        try:
            a = int(input)
            return True
        except :
            return False

    def getDataMean(self, j):
        column_sum = 0

        for i in range(len(self.data)):
            if self.__is_int__(self.data[i][j]):
                column_sum += int(self.data[i][j])

        return str(column_sum / len(self.data))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class TestDataInterpreter():
    def __init__(self, filename = ""):
        self.filename = filename
        self.data = []
        self.processInputFile()

    def processInputFile(self):
        input_data = self.getInputFileContent()
        self.makeDatasetList(input_data)
        
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                self.data[i][j] = float(self.data[i][j])	

    def getInputFileContent(self):
        data_content = []
        
        with open(self.filename, newline='') as csvfile:
            file_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
            
            for row in file_content:
                content_row = []
                for data in row:
                    content_row.append(data)
                data_content.append(content_row)

        return data_content[1:]

    def makeDatasetList(self, input_data):
        for row in input_data:
            self.data.append(row[0].split(','))

        self.patchUnknownData()

    def patchUnknownData(self):
        column_patch_method = ["median", "modus", "modus", "mean", \
        "mean", "modus", "modus", "median", \
        "modus", "median", "modus", "modus", \
        "modus"]

        column_patch_values = self.getColumnPatchVal(column_patch_method)

        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == '?':
                    self.data[i][j] = column_patch_values[j]


    def getColumnPatchVal(self, patch_method):
        patch_val = []

        for i in range(len(patch_method)):
            if patch_method[i] == 'modus':
                patch_val.append(self.getDataModus(i))
            elif patch_method[i] == 'median':
                patch_val.append(self.getDataMedian(i))
            elif patch_method[i] == 'mean':
                patch_val.append(self.getDataMean(i))

        return patch_val

    def getDataModus(self, j):
        data_dict = {}

        for i in range(len(self.data)):
            if str(self.data[i][j]) in data_dict:
                data_dict[str(self.data[i][j])] += 1
            else :
                data_dict[str(self.data[i][j])] = 0

        max_key = ''
        max_val = -1
        for key, val in data_dict.items():
            if val > max_val:
                max_key = key
                max_val = val

        return str(max_val)

    def getDataMedian(self, j):
        column_list = []

        for i in range(len(self.data)):
            column_list.append(self.data[i][j])

        column_list.sort()
        median_idx = (len(column_list)//2) + 1

        return str(column_list[median_idx])

    def __is_int__(self, input):
        try:
            a = int(input)
            return True
        except :
            return False

    def getDataMean(self, j):
        column_sum = 0

        for i in range(len(self.data)):
            if self.__is_int__(self.data[i][j]):
                column_sum += int(self.data[i][j])

        return str(column_sum / len(self.data)) 
#Supporting Function:
import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt ='d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
print("importing sklearn...")
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("importing input interpreter...")
print("importing pickle")
import pickle
import warnings
warnings.filterwarnings('ignore')
print("Done")


inp = InputDataInterpreter(filename="tubes2_HeartDisease_train.csv")
test_data = TestDataInterpreter(filename="tubes2_HeartDisease_test.csv")
train_data,test_data,train_target,test_target=train_test_split(inp.data,inp.target)


print("Training data...")
knn = KNeighborsClassifier(n_neighbors=3)
kNN_model = knn.fit(train_data, train_target)

print("Predicting data...")
knn_y_pred = kNN_model.predict(test_data)

print("Accuracy: %0.2f" % (accuracy_score(test_target,knn_y_pred)*100))
print(classification_report(test_target, knn_y_pred))
kNN_cnf_matrix = confusion_matrix(test_target, knn_y_pred)
plot_confusion_matrix(kNN_cnf_matrix, kNN_model.classes_, title = "kNN Confusion Matrix")
print("Training data...")
gnb = GaussianNB()
naive_model = gnb.fit(train_data, train_target)

print("Predicting data...")
naive_target_pred = naive_model.predict(test_data)

print("Accuracy: %0.2f" % (accuracy_score(test_target,naive_target_pred)*100))

conf_matrix = confusion_matrix(test_target,naive_target_pred)
print(classification_report(test_target, naive_target_pred))
plot_confusion_matrix(conf_matrix, naive_model.classes_, title = "Naive Bayes Confusion Matrix")
print("Training data...")
dt = DecisionTreeClassifier()
dt_model = dt.fit(train_data, train_target)

print("Predicting data...")
target_pred = dt_model.predict(test_data)

print("Accuracy: %0.2f" % (accuracy_score(test_target,target_pred)*100))
print(classification_report(test_target, target_pred))
conf_matrix = (confusion_matrix(test_target,target_pred))
plot_confusion_matrix(conf_matrix, dt_model.classes_, title = "Decision Tree Confusion Matrix")
print("Training data...")
clf = MLPClassifier(max_iter=9000, solver='lbfgs', alpha=1e-5 ,hidden_layer_sizes=(10,9,6), random_state=0, shuffle=True)
clf.fit(train_data, train_target)

print("predicting data...")
target_pred = clf.predict(test_data)

print("Accuracy: %0.2f" % (accuracy_score(test_target,target_pred)*100))
print(classification_report(test_target, target_pred))
conf_matrix = (confusion_matrix(test_target,target_pred))
plot_confusion_matrix(conf_matrix, dt_model.classes_, title = "MLP Confusion Matrix")

