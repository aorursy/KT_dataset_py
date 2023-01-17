# module imports

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import random



# model imports

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



# processing imports

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



print('Welcome!')
# fetch the training file

file_path_20_percent = '../input/nslkdd/KDDTrain+_20Percent.txt'

file_path_full_training_set = '../input/nslkdd/KDDTrain+.txt'

file_path_test = '../input/nslkdd/KDDTest+.txt' 



#df = pd.read_csv(file_path_20_percent)

df = pd.read_csv(file_path_full_training_set)

test_df = pd.read_csv(file_path_test)
# add the column labels

columns = (['duration'

,'protocol_type'

,'service'

,'flag'

,'src_bytes'

,'dst_bytes'

,'land'

,'wrong_fragment'

,'urgent'

,'hot'

,'num_failed_logins'

,'logged_in'

,'num_compromised'

,'root_shell'

,'su_attempted'

,'num_root'

,'num_file_creations'

,'num_shells'

,'num_access_files'

,'num_outbound_cmds'

,'is_host_login'

,'is_guest_login'

,'count'

,'srv_count'

,'serror_rate'

,'srv_serror_rate'

,'rerror_rate'

,'srv_rerror_rate'

,'same_srv_rate'

,'diff_srv_rate'

,'srv_diff_host_rate'

,'dst_host_count'

,'dst_host_srv_count'

,'dst_host_same_srv_rate'

,'dst_host_diff_srv_rate'

,'dst_host_same_src_port_rate'

,'dst_host_srv_diff_host_rate'

,'dst_host_serror_rate'

,'dst_host_srv_serror_rate'

,'dst_host_rerror_rate'

,'dst_host_srv_rerror_rate'

,'attack'

,'level'])



df.columns = columns

test_df.columns = columns



# sanity check

df.head()
# map normal to 0, all attacks to 1

is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)

test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)



#data_with_attack = df.join(is_attack, rsuffix='_flag')

df['attack_flag'] = is_attack

test_df['attack_flag'] = test_attack



# view the result

df.head()
# lists to hold our attack classifications

dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']

probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']

privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']

access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']



# we will use these for plotting below

attack_labels = ['Normal','DoS','Probe','Privilege','Access']



# helper function to pass to data frame mapping

def map_attack(attack):

    if attack in dos_attacks:

        # dos_attacks map to 1

        attack_type = 1

    elif attack in probe_attacks:

        # probe_attacks mapt to 2

        attack_type = 2

    elif attack in privilege_attacks:

        # privilege escalation attacks map to 3

        attack_type = 3

    elif attack in access_attacks:

        # remote access attacks map to 4

        attack_type = 4

    else:

        # normal maps to 0

        attack_type = 0

        

    return attack_type



# map the data and join to the data set

attack_map = df.attack.apply(map_attack)

df['attack_map'] = attack_map



test_attack_map = test_df.attack.apply(map_attack)

test_df['attack_map'] = test_attack_map



# view the result

df.head()
# use a crosstab to get attack vs protocol

attack_vs_protocol = pd.crosstab(df.attack, df.protocol_type)

attack_vs_protocol
# helper function for drawing mulitple charts.

def bake_pies(data_list,labels):

    list_length = len(data_list)

    

    # setup for mapping colors

    color_list = sns.color_palette()

    color_cycle = itertools.cycle(color_list)

    cdict = {}

    

    # build the subplots

    fig, axs = plt.subplots(1, list_length,figsize=(18,10), tight_layout=False)

    plt.subplots_adjust(wspace=1/list_length)

    

    # loop through the data sets and build the charts

    for count, data_set in enumerate(data_list): 

        

        # update our color mapt with new values

        for num, value in enumerate(np.unique(data_set.index)):

            if value not in cdict:

                cdict[value] = next(color_cycle)

       

        # build the wedges

        wedges,texts = axs[count].pie(data_set,

                           colors=[cdict[v] for v in data_set.index])



        # build the legend

        axs[count].legend(wedges, data_set.index,

                           title="Flags",

                           loc="center left",

                           bbox_to_anchor=(1, 0, 0.5, 1))

        # set the title

        axs[count].set_title(labels[count])

        

    return axs   



# get the series for each protocol

icmp_attacks = attack_vs_protocol.icmp

tcp_attacks = attack_vs_protocol.tcp

udp_attacks = attack_vs_protocol.udp



# create the charts

bake_pies([icmp_attacks, tcp_attacks, udp_attacks],['icmp','tcp','udp'])

plt.show()
# get a series with the count of each flag for attack and normal traffic

normal_flags = df.loc[df.attack_flag == 0].flag.value_counts()

attack_flags = df.loc[df.attack_flag == 1].flag.value_counts()



# create the charts

flag_axs = bake_pies([normal_flags, attack_flags], ['normal','attack'])        

plt.show()
# get a series with the count of each service for attack and normal traffic

normal_services = df.loc[df.attack_flag == 0].service.value_counts()

attack_services = df.loc[df.attack_flag == 1].service.value_counts()



# create the charts

service_axs = bake_pies([normal_services, attack_services], ['normal','attack'])        

plt.show()

# get the intial set of encoded features and encode them

features_to_encode = ['protocol_type', 'service', 'flag']

encoded = pd.get_dummies(df[features_to_encode])

test_encoded_base = pd.get_dummies(test_df[features_to_encode])



# not all of the features are in the test set, so we need to account for diffs

test_index = np.arange(len(test_df.index))

column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))



diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)



# we'll also need to reorder the columns to match, so let's get those

column_order = encoded.columns.to_list()



# append the new columns

test_encoded_temp = test_encoded_base.join(diff_df)



# reorder the columns

test_final = test_encoded_temp[column_order].fillna(0)



# get numeric features, we won't worry about encoding these at this point

numeric_features = ['duration', 'src_bytes', 'dst_bytes']



# model to fit/test

to_fit = encoded.join(df[numeric_features])

test_set = test_final.join(test_df[numeric_features])
# create our target classifications

binary_y = df['attack_flag']

multi_y = df['attack_map']



test_binary_y = test_df['attack_flag']

test_multi_y = test_df['attack_map']



# build the training sets

binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y, test_size=0.6)

multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y, test_size = 0.6)
# model for the binary classification

binary_model = RandomForestClassifier()

binary_model.fit(binary_train_X, binary_train_y)

binary_predictions = binary_model.predict(binary_val_X)



# calculate and display our base accuracty

base_rf_score = accuracy_score(binary_predictions,binary_val_y)

base_rf_score
# define the list of models that we want to test

models = [

    RandomForestClassifier(),

    LogisticRegression(max_iter=250),

    KNeighborsClassifier(),

]



# an empty list to capture the performance of each model

model_comps = []



# walk through the models and populate our list

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, binary_train_X, binary_train_y, scoring='accuracy')

    for count, accuracy in enumerate(accuracies):

        model_comps.append((model_name, count, accuracy))
# a box plot will do well to show us overall performance and the variation in the models.

result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])

result_df.pivot(index='count',columns='model_name',values='accuracy').boxplot(rot=45)
# a helper function for getting some analytical data about our predictions

def add_predictions(data_set,predictions,y):

    prediction_series = pd.Series(predictions, index=y.index)



    # we need to add the predicted and actual outcomes to the data

    predicted_vs_actual = data_set.assign(predicted=prediction_series)

    original_data = predicted_vs_actual.assign(actual=y).dropna()

    conf_matrix = confusion_matrix(original_data['actual'], 

                                   original_data['predicted'])

    

    # capture rows with failed predictions

    base_errors = original_data[original_data['actual'] != original_data['predicted']]

    

    # drop columns with no value

    non_zeros = base_errors.loc[:,(base_errors != 0).any(axis=0)]



    # idetify the type of error

    false_positives = non_zeros.loc[non_zeros.actual==0]

    false_negatives = non_zeros.loc[non_zeros.actual==1]



    # put everything into an object

    prediction_data = {'data': original_data,

                       'confusion_matrix': conf_matrix,

                       'errors': base_errors,

                       'non_zeros': non_zeros,

                       'false_positives': false_positives,

                       'false_negatives': false_negatives}

    

    return prediction_data

# capture our prediction data

binary_prediction_data = add_predictions(df,

                                         binary_predictions,

                                         binary_val_y)



# create a heatmap of the confusion matrix

sns.heatmap(data=binary_prediction_data['confusion_matrix'],

            xticklabels = ['Predicted Normal','Predicted Attack'],

            yticklabels = ['Actual Normal','Actual Attack'],

            cmap="YlGnBu",

            fmt='d',

            annot=True)
# dataframe to store incorrect classification

binary_prediction_data['errors'].describe()
# data minus the rows with no variance

binary_prediction_data['non_zeros'].describe()
# see the standard deviation of the false positives

binary_prediction_data['false_positives'].std()
# see the standard deviation of the false negatives

binary_prediction_data['false_negatives'].std()
# distribution of false negatives--what attacks did we miss?

binary_prediction_data['false_negatives'].attack.value_counts().plot.bar()
# we'll need to pull these from the data set

outcomes = ['attack_flag','attack_map', 'actual']



# get the new features we're interested in and drop the outcomes

new_features = (binary_prediction_data['false_positives']==0).all(axis=0)

feature_cols = binary_prediction_data['false_positives'].loc[:,new_features]

feature_cols = feature_cols.drop(outcomes,axis=1)



# Let's get these in a list and take a look

new_feature_columns = list(feature_cols.columns)

new_feature_columns
# add the new freatures

to_fit_new_features = to_fit.join(df[new_feature_columns])



# build the training sets

new_feature_train_X, new_feature_val_X, new_feature_train_y, new_feature_val_y = train_test_split(to_fit_new_features, binary_y)
# model for the binary classification

new_feature_model = RandomForestClassifier()

new_feature_model.fit(new_feature_train_X, new_feature_train_y)

new_feature_predictions = new_feature_model.predict(new_feature_val_X)



# get the score for the model

new_feature_score = accuracy_score(new_feature_predictions,new_feature_val_y)



new_feature_score
# capture the prediction data

new_prediction_data = add_predictions(df,

                                      new_feature_predictions,

                                      new_feature_val_y)



# create a heatmap of the confusion matrix

sns.heatmap(data=new_prediction_data['confusion_matrix'],

            xticklabels = ['Predicted Normal','Predicted Attack'],

            yticklabels = ['Actual Normal','Actual Attack'],

            cmap="YlGnBu",

            fmt='d',

            annot=True)
# distribuition of the false negatives--what attacks did we miss?

new_prediction_data['false_negatives'].attack.value_counts().plot.bar()
# model for the binary classification

full_model = RandomForestClassifier(random_state=1)

full_model.fit(to_fit, binary_y)

full_predictions = full_model.predict(test_set)



# get the score

full_score = accuracy_score(full_predictions,test_binary_y)

full_score
# capture the prediction data

full_prediction_data = add_predictions(test_df, full_predictions, test_binary_y)



# create a heatmap of the confusion matrix

sns.heatmap(data=full_prediction_data['confusion_matrix'],

            xticklabels = ['Predicted Normal','Predicted Attack'],

            yticklabels = ['Actual Normal','Actual Attack'],

            cmap="YlGnBu",

            fmt='d',

            annot=True)
# create our label encoder

label_encoder = LabelEncoder()



# get the intial set of encoded features and encode them

features_to_encode = ['protocol_type', 'flag']

dummy_encoded = pd.get_dummies(df[features_to_encode])

test_dummy_encoded = pd.get_dummies(test_df[features_to_encode])



# now we'll label encode the service column

label_encoder.fit(df.service)

dummy_encoded['service'] = label_encoder.transform(df.service)

test_dummy_encoded['service'] = label_encoder.transform(test_df.service)



# get numeric features, we won't worry about encoding these at this point

numeric_features = ['duration', 'src_bytes', 'dst_bytes']



# model to fit/test

to_fit = dummy_encoded.join(df[numeric_features])

test_set = test_dummy_encoded.join(test_df[numeric_features])



# make sure our columns match

print(to_fit.columns)

print(test_set.columns)
# model for the binary classification

full_model = RandomForestClassifier(random_state=1)

full_model.fit(to_fit, binary_y)

full_predictions = full_model.predict(test_set)



# get the score

full_score = accuracy_score(full_predictions,test_binary_y)

full_score
# add new features

to_fit_new_features = to_fit.join(df[new_feature_columns])

test_set_new_features = test_set.join(test_df[new_feature_columns])
# run the model

full_model.fit(to_fit_new_features,binary_y)

full_predictions = full_model.predict(test_set_new_features)



# get the score

full_score = accuracy_score(full_predictions,test_binary_y)

full_score
# model for the mulit classification

multi_model = RandomForestClassifier()

multi_model.fit(multi_train_X, multi_train_y)

multi_predictions = multi_model.predict(multi_val_X)



# get the score

accuracy_score(multi_predictions,multi_val_y)
# build the training sets

multi_feature_train_X, multi_feature_val_X, multi_feature_train_y, multi_feature_val_y = train_test_split(to_fit_new_features, multi_y)
# model for the mulit classification

multi_model = RandomForestClassifier()

multi_model.fit(multi_feature_train_X, multi_feature_train_y)

multi_predictions = multi_model.predict(multi_feature_val_X)



# get the score

accuracy_score(multi_predictions,multi_feature_val_y)
# capture the prediction data

multi_prediction_data = add_predictions(df, multi_predictions, multi_feature_val_y)



# create a heatmap of the confusion matrix

sns.heatmap(data=multi_prediction_data['confusion_matrix'],

            xticklabels = ['Predicted ' + x for x in attack_labels],

            yticklabels = ['Actual ' + x for x in attack_labels],

            cmap="YlGnBu",

            fmt='d',

            annot=True)
# fit on the full data set

multi_model.fit(to_fit_new_features, multi_y)

full_multi_predictions = multi_model.predict(test_set_new_features)



# get the score

accuracy_score(full_multi_predictions,test_multi_y)
# run the model on the smaller column set

multi_model.fit(to_fit, multi_y)

full_multi_predictions = multi_model.predict(test_set)



# get the score

accuracy_score(full_multi_predictions,test_multi_y)
# build our prediction data

multi_prediction_data = add_predictions(df, full_multi_predictions, test_multi_y)



# create a heatmap of the confusion matrix

sns.heatmap(data=multi_prediction_data['confusion_matrix'],

            xticklabels = ['Predicted ' + x for x in attack_labels],

            yticklabels = ['Actual ' + x for x in attack_labels],

            cmap="YlGnBu",

            fmt='d',

            annot=True)