import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data_root = '../input/'
df = pd.read_csv(data_root+'parkinsons_updrs.data')
print(df.shape)
print(df.columns)
df.head(5)
fig, ax = plt.subplots(1,1)
df["motor_UPDRS"].plot(kind="density")
df["total_UPDRS"].plot(kind="density")
fig.show()
male = len(df[df['sex'] == 0])
female = len(df[df['sex'] == 1])
print("There is {0} males and {1} females.".format(male, female))
def corr_sub_plot(ax, df, title=""):
    corr = df.corr()
    avg_corr = np.absolute(corr.values[np.triu_indices_from(corr.values,1)]).mean()
    ax.set_title(title+" (abs(average)={0:.4})".format(avg_corr))
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xticklabels(df.columns)
    return ax.imshow(corr, interpolation="nearest", cmap='cool', vmin=-1, vmax=1)

fig, ax = plt.subplots()
cax = corr_sub_plot(ax, df.iloc[:,17:], title="Correlation plot ")
fig.colorbar(cax);
from itertools import combinations
def scatter_patient(df, subject_list, columns, patient_filter, scatter_alpha=0.3):
    fig, ax = plt.subplots(figsize=(30,22))
    f = [comb for comb in combinations(range(len(columns)), 2)]
    
    for _, fp, _ in patient_filter:
        fp = fp & subject_list
        
    for i in range(len(f)):
        plt.subplot(5,5,i + 1)
        column_1 = columns[f[i][0]]
        column_2 = columns[f[i][1]]
        
        for name, fp, color in patient_filter:
            plt.scatter(df[fp][column_1], df[fp][column_2], alpha=scatter_alpha, marker='.', color=color, s=5, label=name)
        
        plt.xlabel(column_1)
        plt.ylabel(column_2)
        if(i == 0 or i == len(f)):
            plt.legend(markerscale=5, framealpha=1)


sex_filter_patient = [('Men', df['sex'] == 0, 'red'), 
                      ('Women', df['sex'] == 1, 'black')]
scatter_patient(df, df['subject#'] == df['subject#'], ['NHR', 'HNR', 'PPE', 'DFA', 'RPDE'], sex_filter_patient)
pd.DataFrame(df.age).plot(kind="density");
low_margin = 66
less = df['age'] <= low_margin
more = df['age'] > low_margin

age_filter_patient = [('Age<{}'.format(low_margin), less, 'green'), 
                      ('{}>Age'.format(low_margin), more, 'black')]
scatter_patient(df, True, ['NHR', 'HNR', 'PPE', 'DFA', 'RPDE'], age_filter_patient, scatter_alpha=0.3)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

numerical = ['Jitter(%)', 'Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP',
            'Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'age', 'sex', 'test_time']

features_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))
targets_pipe = make_pipeline(StandardScaler())

X = features_pipe.fit_transform(df[numerical])

targets = df[['motor_UPDRS', 'total_UPDRS']]
y = targets_pipe.fit_transform(targets)

input_width = X.shape[1]
print(input_width)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = df['subject#'], train_size=0.9, random_state=4422)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4422)
import keras
from keras.callbacks import EarlyStopping

from IPython.display import clear_output
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=200, verbose=1, mode='min')

# forked from: gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
class PlotLosses(keras.callbacks.Callback):
    def __init__(self, skip=5, refresh_rate=5, figsize=(17,10), zoom_delta=7):
        self.skip = skip
        self.refresh_rate= refresh_rate
        self.figsize=figsize
        self.fig = plt.figure()
        self.zoom_delta = zoom_delta
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        last_loss = logs.get('loss')
        last_val_loss = logs.get('val_loss')

        self.x.append(self.i)
        self.losses.append(last_loss)
        self.val_losses.append(last_val_loss)
        self.i += 1
        
        if(self.i % self.refresh_rate == 0 and self.i > self.skip):
            clear_output(wait=True)
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(self.x[self.skip:], self.losses[self.skip:], label="loss");
            ax.plot(self.x[self.skip:], self.val_losses[self.skip:], label="val_loss");
            plt.title("{0:.4} loss & {1:.4} validation loss (epoch={2})".format(last_loss, last_val_loss, self.i))
            plt.legend()
            
            if(self.i > 100):
                zoom = min(int(self.i/300) + 1, 4)
                axins = zoomed_inset_axes(ax, zoom, loc=7)
                last_epochs = slice(self.i-self.zoom_delta-1,self.i-1)
                min_y = min(min(self.losses[last_epochs]), min(self.val_losses[last_epochs]))
                max_y =  max(max(self.losses[last_epochs]), max(self.val_losses[last_epochs]))
                if(max_y - min_y < 0.2):
                    max_y += 0.04/zoom
                    min_y -= 0.04/zoom
                
                axins.plot(self.x[self.skip:], self.losses[self.skip:])
                axins.plot(self.x[self.skip:], self.val_losses[self.skip:])
                axins.set_xlim(last_epochs.start, last_epochs.stop)
                axins.set_ylim(min_y, max_y)
                mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

            plt.show()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization

plot_losses = PlotLosses()

def make_fully_connected_regressor(neuron_per_layers, input_shape):
    model = Sequential([
        Dense(neuron_per_layers, input_shape=input_shape, kernel_initializer='he_uniform', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(neuron_per_layers, kernel_initializer='he_uniform', activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(neuron_per_layers, kernel_initializer='he_uniform', activation='relu'),
        BatchNormalization(),
        Dense(2, kernel_initializer='he_uniform', activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = make_fully_connected_regressor(neuron_per_layers=105, input_shape=(input_width,))

model.fit(X_train, y_train, epochs=2000, batch_size=256, verbose=0, validation_data=(X_val, y_val), callbacks=[earlystop, plot_losses])
model.save('model_1.h5')
from sklearn.metrics import mean_squared_error

test_predictions = model.predict(X_test)
inversed_test_labels = targets_pipe.inverse_transform(y_test)
inversed_predictions = targets_pipe.inverse_transform(test_predictions)

motor_UPDRS_se = mean_squared_error(inversed_test_labels[:,0], inversed_predictions[:,0])
total_UPDRS_se = mean_squared_error(inversed_test_labels[:,1], inversed_predictions[:,1])

motor_UPDRS_se, total_UPDRS_se
transforms = [np.exp, np.log, np.tanh, np.power, np.sqrt]

for e in numerical:
    ref_motor = abs(np.corrcoef(df[e], df['motor_UPDRS'])).mean()
    ref_total = abs(np.corrcoef(df[e], df['total_UPDRS'])).mean()
    print("Current column={0}".format(e))
    
    for t in transforms:
        transformed = 0
        if t is np.power:
            transformed = t(df[e],2)
        else:
            transformed = t(df[e])
            
        motor = abs(np.corrcoef(transformed, df['motor_UPDRS'])).mean()
        total = abs(np.corrcoef(transformed, df['total_UPDRS'])).mean()
        
        if(motor >= ref_motor + 0.01):
            diff = motor - ref_motor
            print("transformer={0} enhance correlation for motor_UPDRS (+{1:.4} +{2:.4})".format(t, motor - ref_motor, ((ref_motor+diff)/ref_motor - 1)*100))
        if(total >= ref_total + 0.01):
            diff = total - ref_total
            print("transformer={0} enhance correlation for total_UPDRS (+{1:.4} +{2:.4}%)".format(t, total - ref_total, ((ref_total+diff)/ref_total - 1)*100))
to_log = ['Jitter(%)', 'Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR']
for feature in to_log:
    df[feature+'_log'] = np.log(df[feature])
    
df['HNR_sq'] = np.power(df['HNR'],2)
numerical_v2 = ['Jitter(%)_log', 'Jitter(Abs)_log','Jitter:RAP_log','Jitter:PPQ5_log','Jitter:DDP_log',
            'Shimmer_log','Shimmer(dB)_log','Shimmer:APQ3_log','Shimmer:APQ5_log','Shimmer:APQ11_log','Shimmer:DDA_log',
            'NHR_log', 'HNR_sq'] + numerical

features_pipe_v2 = make_pipeline(StandardScaler(), PCA(n_components=0.96))
targets_pipe_v2 = make_pipeline(StandardScaler())

X2 = features_pipe_v2.fit_transform(df[numerical_v2])
y2 = targets_pipe_v2.fit_transform(targets)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=0.9, stratify = df['subject#'], random_state=4422)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2, random_state=4422)
input_width2 = X2.shape[1]
print(input_width2)
augmented_model = make_fully_connected_regressor(neuron_per_layers=105, input_shape=(input_width2,))

augmented_model.fit(X2_train, y2_train, epochs=2000, batch_size=256, verbose=0, validation_data=(X2_val, y2_val), callbacks=[earlystop, plot_losses])
test_predictions = augmented_model.predict(X2_test)
inversed_test_labels = targets_pipe_v2.inverse_transform(y2_test)
inversed_predictions = targets_pipe_v2.inverse_transform(test_predictions)

motor_UPDRS_se = mean_squared_error(inversed_test_labels[:,0], inversed_predictions[:,0])
total_UPDRS_se = mean_squared_error(inversed_test_labels[:,1], inversed_predictions[:,1])

motor_UPDRS_se, total_UPDRS_se

