import pandas as pd
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


from keras.layers import Input, Dense, Dropout
from keras.models import Model
path = '../input/adult.csv'
input_data = pd.read_csv(path, na_values="?")
input_data.head()
input_data = input_data[input_data['race'].isin(['White', 'Black'])]
input_data.head()
# sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
sensitive_attribs = ['race', 'gender']
A = input_data[sensitive_attribs]
A = pd.get_dummies(A,drop_first=True)
A.columns = sensitive_attribs
A.head()
y = (input_data['income'] == '>50K').astype(int)
X = input_data.drop(labels=['income', 'race', 'gender'],axis=1)

X = X.fillna('Unknown')

X = pd.get_dummies(X, drop_first=True)
# split into train/test set
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.5, 
                                                                     stratify=y, random_state=7)
# standardize the data
scaler = StandardScaler().fit(X_train)
#scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
def p_rule(y_pred, a_values, threshold=0.5):
    y_a_1 = y_pred[a_values == 1] > threshold if threshold else y_pred[a_values == 1]
    y_a_0 = y_pred[a_values == 0] > threshold if threshold else y_pred[a_values == 0]
    odds = y_a_1.mean() / y_a_0.mean()
    return np.min([odds, 1/odds]) * 100
def make_trainable_fn(net): # Produces a function that makes a network trainable or not
    def make_trainable(flag): # Loop over layers and set their trainability
        net.trainable = flag
        # We need to not only change the model flag but also the layer flags: https://github.com/keras-team/keras/issues/4674
        for layer in net.layers:
            layer.trainable = flag
    return make_trainable
def compute_class_weights(data_set):
    class_values = [0, 1]
    class_weights = []
    if len(data_set.shape) == 1:
        balanced_weights = compute_class_weight('balanced', class_values, data_set)
        class_weights.append(dict(zip(class_values, balanced_weights)))
    else:
        n_attr =  data_set.shape[1]
        for attr_idx in range(n_attr):
            balanced_weights = compute_class_weight('balanced', class_values,
                                                    np.array(data_set)[:,attr_idx])
            class_weights.append(dict(zip(class_values, balanced_weights)))
    return class_weights
def compute_target_class_weights(y):
    class_values  = [0,1]
    balanced_weights =  compute_class_weight('balanced', class_values, y)
    class_weights = {'y': dict(zip(class_values, balanced_weights))}
    return class_weights
n_features=X_train.shape[1]
n_sensitive=A_train.shape[1]
lambdas=[130., 30.]

clf_inputs = Input(shape=(n_features,)) # Classifier input = All features

############### Create CLF net ########################
x = Dense(32, activation='relu')(clf_inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid', name='y')(x)
clf_net = Model(inputs=[clf_inputs], outputs=[outputs])
#######################################################
adv_inputs = Input(shape=(1,)) # Adversary input = Classifier output (one number)

############## Create ADV net #########################
x = Dense(32, activation='relu')(adv_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = [Dense(1, activation='sigmoid')(x) for _ in range(n_sensitive)]
adv_net = Model(inputs=[adv_inputs], outputs=outputs)
#######################################################
############## Create train switches #################
trainable_clf_net = make_trainable_fn(clf_net) # Get function to make classifier trainable

trainable_adv_net = make_trainable_fn(adv_net) # Function to make adversary trainable

######################################################
#clf = compile_clf(clf_net) # Compile classifier

clf = clf_net
trainable_clf_net(True)
clf.compile(loss='binary_crossentropy', optimizer='adam')

# Creates a classifier adversary super net
adv_out = adv_net(clf_net(clf_inputs))
clf_out = clf_net(clf_inputs)
clf_w_adv = Model(inputs=[clf_inputs], outputs=[clf_out]+adv_out)

# The adversary is not trainable the classifier is
trainable_clf_net(True)
trainable_adv_net(False)
# Create a weighted loss for all sensitive variables
loss_weights = [1.]+[-lambda_param for lambda_param in lambdas]
# Compile super net
clf_w_adv.compile(loss='binary_crossentropy', 
                  loss_weights=loss_weights,
                  optimizer='adam')
# Compile adversary with the classifier as inputs
adv = Model(inputs=[clf_inputs], outputs=adv_net(clf_net(clf_inputs)))
# Classifier is not trainable, adversary is
trainable_clf_net(False)
trainable_adv_net(True)
adv.compile(loss='binary_crossentropy', optimizer='adam')
trainable_clf_net(True)
clf.fit(X_train.values, y_train.values, epochs=10)
trainable_clf_net(False)
trainable_adv_net(True)
class_weight_adv = compute_class_weights(A_train)
adv.fit(X_train.values, np.hsplit(A_train.values, A_train.shape[1]), class_weight=class_weight_adv,epochs=10)
y_pred = clf.predict(X_test)
for sens in A_test.columns:
    pr = p_rule(y_pred,A_test[sens])
    print(sens,pr)
acc = accuracy_score(y_test,(y_pred>0.5))* 100
print('Clf acc: {:.2f}'.format(acc))
n_iter=250
batch_size=128
n_sensitive = A_train.shape[1]

class_weight_clf_w_adv = [{0:1., 1:1.}]+class_weight_adv

val_metrics = pd.DataFrame()

fairness_metrics = pd.DataFrame()

for idx in range(n_iter): # Train for n epochs

    # train adverserial
    trainable_clf_net(False)
    trainable_adv_net(True)
    adv.fit(X_train.values, 
            np.hsplit(A_train.values, A_train.shape[1]), 
            batch_size=batch_size, 
            class_weight=class_weight_adv, 
            epochs=1, verbose=0)


    # train classifier
    # Make classifier trainable and adversery untrainable
    trainable_clf_net(True)
    trainable_adv_net(False)
    # Sample batch
    indices = np.random.permutation(len(X_train))[:batch_size]
    # Train on batch
    clf_w_adv.train_on_batch(X_train.values[indices], 
                            [y_train.values[indices]]+np.hsplit(A_train.values[indices], n_sensitive),
                            class_weight=class_weight_clf_w_adv)

    
    # Make validation data predictions
    y_pred = pd.Series(clf.predict(X_test).ravel(), index=y_test.index)

    roc_auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred>0.5))*100
    # Calculate ROC and accuracy
    val_metrics.loc[idx, 'ROC AUC'] = roc_auc
    val_metrics.loc[idx, 'Accuracy'] = acc

    # Calculate p rule
    for sensitive_attr in A_test.columns:
        fairness_metrics.loc[idx, sensitive_attr] = p_rule(y_pred,A_test[sensitive_attr])

    print('Epoch: {}, Accuracy: {:.2f}, Race P: {:.2f}, Gender P: {:.2f}'.format(idx,
                                                                                 acc, 
                                                                                 fairness_metrics.loc[idx, 'race'],
                                                                                 fairness_metrics.loc[idx, 'gender']))


# adverserial train on train set and validate on test set
#vm, fm = fit(X_train, y_train, A_train,validation_data=(X_test, y_test, A_test),n_iter=200)
plt.figure(figsize=(10,7))
plt.xlabel('Epochs')
plt.plot(val_metrics['Accuracy'],label='Accuracy')
plt.plot(val_metrics['ROC AUC']*100,label='ROC AUC')
plt.plot(fairness_metrics['race'],label='Race')
plt.plot(fairness_metrics['gender'],label='Gender')
plt.legend()
