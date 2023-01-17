import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import sys



random.seed(0)

train_rawDf = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')

test_rawDf = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/test_data.csv')

train_rawDf.head()
train_rawDf.describe().T
train_rawDf.info()
city_code_patient_mode = train_rawDf['City_Code_Patient'].mode()[0]

train_Df = train_rawDf.copy()

test_Df = test_rawDf.copy()

train_Df['Bed Grade'] = train_Df['Bed Grade'].fillna(3.0)

train_Df['City_Code_Patient'] = train_Df['City_Code_Patient'].fillna(city_code_patient_mode)

test_Df['Bed Grade'] = test_Df['Bed Grade'].fillna(3.0)

test_Df['City_Code_Patient'] = test_Df['City_Code_Patient'].fillna(city_code_patient_mode)

train_Df.info()
categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age', 'Stay']

categorical_column_name_to_value_to_integer_dict = {}

for column_name in categorical_columns:

    value_to_integer_dict = {}

    values = sorted(train_Df[column_name].unique())

    for index, value in enumerate(values):

        value_to_integer_dict[value] = index

    categorical_column_name_to_value_to_integer_dict[column_name] = value_to_integer_dict

    

# Manually set values for 'Severity of Illness' to have an ordinal interpretation: Minor < Moderate < Extreme

categorical_column_name_to_value_to_integer_dict['Severity of Illness'] = {'Minor': 0, 'Moderate': 1, 'Extreme': 2}

print (categorical_column_name_to_value_to_integer_dict)
def ReplaceCategoricalValues(dataframe, categorical_column_name_to_value_to_integer_dict):

    for column_name in dataframe.columns:

        if column_name in categorical_column_name_to_value_to_integer_dict:

            old_values = list(categorical_column_name_to_value_to_integer_dict[column_name].keys())

            new_values = list(categorical_column_name_to_value_to_integer_dict[column_name].values())

            dataframe[column_name] = dataframe[column_name].replace(old_values, new_values)



ReplaceCategoricalValues(train_Df, categorical_column_name_to_value_to_integer_dict)

ReplaceCategoricalValues(test_Df, categorical_column_name_to_value_to_integer_dict)

train_Df.head()
train_Df = train_Df.drop(columns=['case_id', 'patientid'])

test_Df = test_Df.drop(columns=['patientid']) # We'll need 'case_id' for submission

train_Df.head()
continuous_values_columns = ['Available Extra Rooms in Hospital', 'Bed Grade', 'Severity of Illness', 'Visitors with Patient', 'Age', 'Admission_Deposit', 'Stay']

continuous_column_to_min_max_dict = {column_name: (train_Df[column_name].min(), train_Df[column_name].max()) for column_name in continuous_values_columns}

for column_name in continuous_values_columns:

    min_value = continuous_column_to_min_max_dict[column_name][0]

    max_value = continuous_column_to_min_max_dict[column_name][1]

    if min_value == max_value:

        raise ValueError("min_value == max_value ({})".format(min_value))

    train_Df[column_name] = (train_Df[column_name] - min_value)/(max_value - min_value)

    if column_name is not 'Stay':

        test_Df[column_name] = (test_Df[column_name] - min_value)/(max_value - min_value)
train_Df['City_Code_Patient'] = train_Df['City_Code_Patient'].astype(int)

test_Df['City_Code_Patient'] = test_Df['City_Code_Patient'].astype(int)
train_Df.head()
train_target_Df = train_Df['Stay']

train_features_Df = train_Df.drop(columns=['Stay'])

train_features_Df.head()
from sklearn.model_selection import train_test_split

print("len(train_features_Df) = {}".format(len(train_features_Df)))

validation_proportion = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(train_features_Df, train_target_Df, 

                                                     test_size=validation_proportion)

print("len(X_train) = {}".format(len(X_train)))

print ("len(X_valid) = {}".format(len(X_valid)))
integer_to_letter_dict = {}

for i in range(26):

    integer_to_letter_dict[i] = chr(i + 97)

for i in range(26, 40):

    integer_to_letter_dict[i] = 'A{}'.format(chr(i - 26 + 97))

print (integer_to_letter_dict)

classification_tree_categorical_columns = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'City_Code_Patient', 'Type of Admission']

classification_tree_categorical_column_name_to_value_to_integer_dict = {column_name: integer_to_letter_dict for column_name in classification_tree_categorical_columns}

classification_tree_X_train = X_train.copy()

ReplaceCategoricalValues(classification_tree_X_train, classification_tree_categorical_column_name_to_value_to_integer_dict)

classification_tree_X_valid = X_valid.copy()

ReplaceCategoricalValues(classification_tree_X_valid, classification_tree_categorical_column_name_to_value_to_integer_dict)

classification_tree_X_test = test_Df.copy()

ReplaceCategoricalValues(classification_tree_X_test, classification_tree_categorical_column_name_to_value_to_integer_dict)



classification_tree_X_train.head()
classification_tree_X_train = pd.get_dummies(classification_tree_X_train[classification_tree_X_train.columns], drop_first=True) # Will split the categorical columns (identified by their alphabetic values) into multiple binary columns, encoding them as one-hot

classification_tree_X_valid = pd.get_dummies(classification_tree_X_valid[classification_tree_X_valid.columns], drop_first=True)

classification_tree_X_test = pd.get_dummies(classification_tree_X_test[classification_tree_X_test.columns], drop_first=True)

# Make sur both feature matrices have all the column names

column_names_1 = classification_tree_X_train.columns

column_names_2 = classification_tree_X_valid.columns

column_names_test = classification_tree_X_test.columns

column_names_union = set(column_names_1).union(set(column_names_2))

for column_name in column_names_union:

    if column_name not in column_names_1:

        classification_tree_X_train[column_name] = [0] * len(classification_tree_X_train.index)

    if column_name not in column_names_2:

        classification_tree_X_valid[column_name] = [0] * len(classification_tree_X_valid.index)

    if column_name not in column_names_test:

        classification_tree_X_test[column_name] = [0] * len(classification_tree_X_test.index)

    

    

# Sort the column names in alphabetic order, so are identically ordered for both dataframes

classification_tree_X_train = classification_tree_X_train.reindex(sorted(classification_tree_X_train.columns), axis=1)

classification_tree_X_valid = classification_tree_X_valid.reindex(sorted(classification_tree_X_valid.columns), axis=1)

classification_tree_X_test = classification_tree_X_test.reindex(sorted(classification_tree_X_test.columns), axis=1)

classification_tree_X_train.head()
print(classification_tree_X_train.columns)

print(classification_tree_X_valid.columns)
def ClassFromPrediction(prediction_continuous):

    if prediction_continuous < 0.05:

        return 0

    elif prediction_continuous < 0.15:

        return 1

    elif prediction_continuous < 0.25:

        return 2

    elif prediction_continuous < 0.35:

        return 3

    elif prediction_continuous < 0.45:

        return 4

    elif prediction_continuous < 0.55:

        return 5

    elif prediction_continuous < 0.65:

        return 6

    elif prediction_continuous < 0.75:

        return 7

    elif prediction_continuous < 0.85:

        return 8

    elif prediction_continuous < 0.95:

        return 9

    else:

        return 10

    from sklearn import tree

    

y_train_classes = [ClassFromPrediction(y) for y in y_train]



from sklearn import tree



classification_tree = tree.DecisionTreeClassifier()

classification_tree.fit(classification_tree_X_train, y_train_classes)

y_valid_classes = [ClassFromPrediction(y) for y in y_valid]



classification_tree_predicted_validation_y = classification_tree.predict(classification_tree_X_valid)

print ("len(classification_tree_predicted_validation_y) = {}".format(len(classification_tree_predicted_validation_y)))

print("len(y_valid_classes) = {}".format(len(y_valid_classes)))
classification_tree_prediction_mtx = np.zeros((11, 11), dtype=int)

for prediction, target in zip(list(classification_tree_predicted_validation_y), list(y_valid_classes)):

    classification_tree_prediction_mtx[target, prediction] += 1





import seaborn as sns

ax = sns.heatmap(classification_tree_prediction_mtx)

correct = (classification_tree_predicted_validation_y == y_valid_classes)

accuracy = correct.sum() / correct.size

print ("classification tree accuracy = {}".format(accuracy))
y_train_class_to_occurrences_dict = {classNdx: 0 for classNdx in range(11)}

for target_obs in y_train_classes:

    y_train_class_to_occurrences_dict[target_obs] += 1

print (y_train_class_to_occurrences_dict)

y_train_mode = 0

highest_count = -1

for classNdx, count in y_train_class_to_occurrences_dict.items():

    if count > highest_count:

        highest_count = count

        y_train_mode = classNdx



print ("y_train_mode = {}".format(y_train_mode))

correct_dumb_predictions = 0

for valid_obs in y_valid_classes:

    if valid_obs == y_train_mode:

        correct_dumb_predictions += 1

dumb_accuracy = correct_dumb_predictions/len(y_valid_classes)

print ("Dumb accuracy obtained when always predicting {}: {}".format(y_train_mode, dumb_accuracy))
from catboost import Pool, CatBoostClassifier

cat_boost_train_dataset = Pool(data=classification_tree_X_train, label=y_train_classes)

cat_boost_valid_dataset = Pool(data=classification_tree_X_valid, label=y_valid_classes)

# Initialising catboost classifier



cat_boost_model = CatBoostClassifier(eval_metric='Accuracy')

    
cat_boost_model.fit(cat_boost_train_dataset, eval_set=cat_boost_valid_dataset)
cat_boost_predicted_validation_y = cat_boost_model.predict(classification_tree_X_valid)

cat_boost_prediction_mtx = np.zeros((11, 11), dtype=int)

for prediction, target in zip(list(cat_boost_predicted_validation_y), list(y_valid_classes)):

    cat_boost_prediction_mtx[target, prediction] += 1





ax = sns.heatmap(cat_boost_prediction_mtx)
correct = [cat_boost_predicted_validation_y[i, 0] == y_valid_classes[i] for i in range(len(y_valid_classes))]

correct_sum = sum([1 for v in correct if v == True ])

accuracy = correct_sum / len(correct)

print ("cat_boost accuracy = {}".format(accuracy))
import torch



classification_X_train_Tsr = torch.tensor(classification_tree_X_train.values)

classification_X_valid_Tsr = torch.tensor(classification_tree_X_valid.values)

y_train_classes_Tsr = torch.tensor(y_train_classes).long()

y_valid_classes_Tsr = torch.tensor(y_valid_classes).long()
class MLP(torch.nn.Module):

    def __init__(self, 

                 number_of_inputs,

                 number_of_classes,

                 hidden_layer_width=128,

                 dropout_proportion=0.5):

        super(MLP, self).__init__()

        self.number_of_inputs = number_of_inputs

        self.number_of_classes = number_of_classes

        self.hidden_layer_width = hidden_layer_width

        self.linear1 = torch.nn.Linear(self.number_of_inputs, self.hidden_layer_width)

        self.linear2 = torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width)

        self.linear3 = torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width)

        self.linear4 = torch.nn.Linear(self.hidden_layer_width, self.number_of_classes)

        self.dropout = torch.nn.Dropout(p=dropout_proportion)

        self.batchnorm = torch.nn.BatchNorm1d(self.hidden_layer_width)

        

    def forward(self, inputTsr):

        # inputTsr.shape = (N, self._number_of_inputs)

        latent1Tsr = torch.nn.functional.relu(self.linear1(inputTsr)) # latent1Tsr.shape = (N, first_layer_width)

        latent1Tsr = self.dropout(latent1Tsr)

        #latent1Tsr = self.batchnorm(latent1Tsr)

        latent2Tsr = torch.nn.functional.relu( self.linear2(latent1Tsr))

        #latent2Tsr = self.batchnorm(latent2Tsr)

        latent3Tsr = torch.nn.functional.relu( self.linear3(latent2Tsr))

        latent3Tsr = self.batchnorm(latent3Tsr)

        outputTsr = self.linear4(latent3Tsr)

        return outputTsr



number_of_classes = 11

mlp = MLP(classification_X_train_Tsr.shape[1], number_of_classes)
from torch.utils.data import Dataset, DataLoader

class ClassificationObservationDataset(Dataset):

    def __init__(self, featuresTsr, target_class_Tsr):

        if featuresTsr.shape[0] != target_class_Tsr.shape[0]:

            raise ValueError("ClassificationObservationDataset.__init__(): featuresTsr.shape[0] ({}) != target_class_Tsr.shape[0] ({})".format(featuresTsr.shape[0], target_class_Tsr.shape[0]))

        self.featuresTsr = featuresTsr

        self.target_class_Tsr = target_class_Tsr

        

    def __len__(self):

        return self.featuresTsr.shape[0]

    

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.featuresTsr.shape[0]:

            raise IndexError("ClassificationObservationDataset.__getitem__(): idx ({}) is out of [0, {}]".format(self.featuresTsr.shape[0] - 1))

        return (self.featuresTsr[idx].float(), self.target_class_Tsr[idx])

    

train_dataset = ClassificationObservationDataset(classification_X_train_Tsr, y_train_classes_Tsr)

validation_dataset = ClassificationObservationDataset(classification_X_valid_Tsr, y_valid_classes_Tsr)
print(y_train_class_to_occurrences_dict)

#occurrences_sum = sum([v for k, v in y_train_class_to_occurrences_dict.items()])

class_weight_Tsr = torch.zeros(len(y_train_class_to_occurrences_dict))

for classNdx, occurrences in y_train_class_to_occurrences_dict.items():

    class_weight = 1.0/max(occurrences, 1)

    class_weight_Tsr[classNdx] = class_weight

class_weight_sum = class_weight_Tsr.sum().item()

class_weight_Tsr = class_weight_Tsr / class_weight_sum



print (class_weight_Tsr)
mlp_parameters = filter(lambda p: p.requires_grad, mlp.parameters())

optimizer = torch.optim.Adam(mlp_parameters, lr=0.0001)

lossFcn = torch.nn.CrossEntropyLoss()#weight=class_weight_Tsr)

train_dataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataLoader = DataLoader(validation_dataset, batch_size=len(validation_dataset))

useCuda = torch.cuda.is_available()

if useCuda:

    mlp = mlp.cuda()

lowest_validation_loss = sys.float_info.max

champion_classification_mlp_filepath = '/kaggle/working/classification_mlp.pth'

number_of_epochs = 20

for epoch in range(1, number_of_epochs):

    mlp.train()

    loss_sum = 0.0

    

    number_of_batches = 0

    for (featuresTsr, target_class_ndx) in train_dataLoader:

        if number_of_batches % 20 == 1:

            print (".", end="", flush=True)

        if useCuda:

            featuresTsr = featuresTsr.cuda()

            target_class_ndx = target_class_ndx.cuda()

        predicted_class_ndx = mlp(featuresTsr)

        optimizer.zero_grad()

        loss = lossFcn(predicted_class_ndx, target_class_ndx)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        number_of_batches += 1

    train_loss = loss_sum/number_of_batches

    print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))

    

    # Validation

    mlp.eval()

    with torch.set_grad_enabled(False):

        for validation_features_Tsr, validation_target_class_ndx_Tsr in valid_dataLoader: # Will be a single pass since batch_size=len(validation_dataset)

            if useCuda:

                validation_features_Tsr = validation_features_Tsr.cuda()

                validation_target_class_ndx_Tsr = validation_target_class_ndx_Tsr.cuda()

            validation_predicted_class_ndx_Tsr = mlp(validation_features_Tsr)

            validation_loss = lossFcn(validation_predicted_class_ndx_Tsr, validation_target_class_ndx_Tsr).item()

            if validation_loss < lowest_validation_loss:

                lowest_validation_loss = validation_loss

                torch.save(mlp.state_dict(), champion_classification_mlp_filepath)

            # Validation accuracy

            validation_correct_predictions = (validation_predicted_class_ndx_Tsr.argmax(dim=1) == validation_target_class_ndx_Tsr).sum().item()

            validation_accuracy = validation_correct_predictions / validation_target_class_ndx_Tsr.shape[0]

            print ("validation_loss = {}; validation_accuracy = {}".format(validation_loss, validation_accuracy))

            
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(max_depth=16, random_state=0)

random_forest_classifier.fit(classification_tree_X_train, y_train_classes)
random_forest_predicted_validation_y = random_forest_classifier.predict(classification_tree_X_valid)

random_forest_confusion_mtx = np.zeros((11, 11), dtype=int)

for prediction, target in zip(list(random_forest_predicted_validation_y), list(y_valid_classes)):

    random_forest_confusion_mtx[target, prediction] += 1





ax = sns.heatmap(random_forest_confusion_mtx)
correct = (random_forest_predicted_validation_y == y_valid_classes)

accuracy = correct.sum() / correct.size

print ("random_forest accuracy = {}".format(accuracy))
classification_tree_X_test.columns
# CatBoost classifier

print (classification_tree_X_test.columns)

cat_boost_prediction = cat_boost_model.predict(classification_tree_X_test.drop(columns=['case_id']))

print ("cat_boost_prediction.shape = {}".format(cat_boost_prediction.shape))



# Neural network classifier

classification_X_test_Tsr = torch.tensor(classification_tree_X_test.drop(columns=['case_id']).values).float()

mlp.load_state_dict(torch.load(champion_classification_mlp_filepath))

mlp.eval()

mlp_prediction_Tsr = torch.argmax(mlp(classification_X_test_Tsr), dim=1)

print ("mlp_prediction_Tsr.shape = {}".format(mlp_prediction_Tsr.shape))



# Random forest

random_forest_prediction = random_forest_classifier.predict(classification_tree_X_test.drop(columns=['case_id']))

print("random_forest_prediction.shape = {}".format(random_forest_prediction.shape))



number_of_unanimities = 0

number_of_two_votes = 0

number_of_draws = 0

case_id_to_prediction_dict = {}



for testNdx, row in classification_tree_X_test.iterrows():

    case_id = int(row['case_id'])

    cat_boost_predicted_class = cat_boost_prediction[testNdx, 0]

    neural_network_predicted_class = mlp_prediction_Tsr[testNdx].item()

    random_forest_predicted_class = random_forest_prediction[testNdx]

    #print("{}: {}, {}, {}".format(case_id, cat_boost_predicted_class, neural_network_predicted_class, random_forest_predicted_class))

    

    if cat_boost_predicted_class == neural_network_predicted_class and cat_boost_predicted_class == random_forest_predicted_class:

        case_id_to_prediction_dict[case_id] = cat_boost_predicted_class

        number_of_unanimities += 1

    elif cat_boost_predicted_class == neural_network_predicted_class:

        case_id_to_prediction_dict[case_id] = cat_boost_predicted_class

        number_of_two_votes += 1

    elif neural_network_predicted_class == random_forest_predicted_class:

        case_id_to_prediction_dict[case_id] = neural_network_predicted_class

        number_of_two_votes += 1

    elif cat_boost_predicted_class == random_forest_predicted_class:

        case_id_to_prediction_dict[case_id] = cat_boost_predicted_class

        number_of_two_votes += 1

    else:

        case_id_to_prediction_dict[case_id] = cat_boost_predicted_class

        number_of_draws += 1

        

print("number_of_unanimities = {}; number_of_two_votes = {}; number_of_draws = {}".format(number_of_unanimities, number_of_two_votes, number_of_draws))

        

print("case_id_to_prediction_dict = {}".format(case_id_to_prediction_dict))



    
# Write a submission file

with open('/kaggle/working/submission.csv', 'w+') as submission_file:

    submission_file.write('case_id,Stay\n')

    for case_id, prediction in case_id_to_prediction_dict.items():

        submission_file.write('{},{}\n'.format(case_id, prediction))