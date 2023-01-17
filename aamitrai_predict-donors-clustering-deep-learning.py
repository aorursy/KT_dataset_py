# General libraries
import os
from collections import Counter
import warnings

# Data analysis and preparation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import cufflinks as cf

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from keras import optimizers, initializers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Set configuration
cf.go_offline()
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 300)
%%time
input_dir = '../input/'
all_projects = pd.read_csv(input_dir + 'Projects.csv')
all_donations = pd.read_csv(input_dir + 'Donations.csv')
all_donors = pd.read_csv(input_dir + 'Donors.csv')
all_schools = pd.read_csv(input_dir + 'Schools.csv')
all_resources = pd.read_csv(input_dir + 'Resources.csv')
all_teachers = pd.read_csv(input_dir + 'Teachers.csv')
# Number of donors for building model with a subset
donor_count = 50000 

# Number of segments the donors needs to be split into
# One should choose the number of segments based on the sample size and volume of donors who need to be targeted.
n_donor_segments = 20
don_count = (all_donations.groupby('Donor ID')
             .size()
             .to_frame(name='Number of Donations')
             .reset_index()
            )

# Select target donors
target_donors = don_count[don_count['Number of Donations'] <= 5]
target_donations = all_donations[all_donations['Donor ID'].isin(target_donors['Donor ID'])]
target_donations = target_donations.sort_values('Donation Received Date', ascending=False)
target_donors = target_donations.drop_duplicates('Donor ID', keep='first')['Donor ID']
target_donors = target_donors.head(donor_count).to_frame(name='Donor ID').reset_index()

# Select target projects
target_projects = target_donors.merge(target_donations, on='Donor ID')
target_projects = target_projects['Project ID'].unique()
target_projects = all_projects[all_projects['Project ID'].isin(target_projects)]

# Select target donation
target_donations = target_donations[target_donations['Donor ID'].isin(target_donors['Donor ID'].values)]
# merged donations
merged_donation = target_donations.merge(target_projects, on='Project ID')
merged_donation = merged_donation.merge(all_donors, on='Donor ID')
merged_donation = merged_donation.merge(all_schools, on='School ID')
merged_donation.shape
donation_cols = ['Project ID', 'Donor ID', 'Donation Amount', 'Project Subject Category Tree',
                'Project Subject Subcategory Tree', 'Project Grade Level Category', 'Project Resource Category',
                'Project Cost', 'Donor State', 'Donor Is Teacher', 'School Metro Type', 'School Percentage Free Lunch',
                'School State']

donation_master = (merged_donation.groupby(['Project ID', 'Donor ID'])
                   .agg({'Donation Amount':'sum'})
                   .rename(columns={'Donation Amount': 'Total Donation'})
                   .reset_index()
                  )

donation_master = merged_donation[donation_cols].merge(donation_master, on=['Project ID', 'Donor ID'])
donation_master = (donation_master.drop_duplicates(['Project ID', 'Donor ID'], keep='first')
                   .drop('Donation Amount', axis=1)
                   .rename(columns={'Total Donation':'Donation Amount'})
                  )
donation_master.head()
# Project Category and Subcategory are stacked columns. A classromm request can span across multiple categories.
# I will start by exploding the columns and then analyze the trend over the years
def stack_attributes(df, target_column, separator=', '):
    df = df.dropna(subset=[target_column])
    df = (df.set_index(df.columns.drop(target_column,1).tolist())
          [target_column].str.split(separator, expand=True)
          .stack().str.strip()
          .reset_index()
          .rename(columns={0:target_column})
          .loc[:, df.columns])
    df = (df.groupby([target_column, 'Project Posted Date'])
          .size()
          .to_frame(name ='Count')
          .reset_index())
    
    return df

def plot_trend(df, target_column, chartType=go.Scatter,
              datecol='Project Posted Date', 
              ytitle='Number of relevant classroom requests'):
    trend = []
    for category in list(df[target_column].unique()):
        temp = chartType(
            x = df[df[target_column]==category][datecol],
            y = df[df[target_column]==category]['Count'],
            name=category
        )
        trend.append(temp)
    
    layout = go.Layout(
        title = 'Trend of ' + target_column,
        xaxis=dict(
            title='Year & Month',
            zeroline=False,
        ),
        yaxis=dict(
            title=ytitle,
        ),
    )
    
    fig = go.Figure(data=trend, layout=layout)
    iplot(fig)
    
proj = all_projects[['Project Subject Category Tree',
                     'Project Subject Subcategory Tree',
                     'Project Resource Category',
                     'Project Grade Level Category',
                     'Project Posted Date']].copy()
proj['Project Posted Date'] = all_projects['Project Posted Date'].str.slice(start=0, stop=4)

proj_cat = stack_attributes (proj, 'Project Subject Category Tree')
proj_sub_cat = stack_attributes (proj, 'Project Subject Subcategory Tree')
proj_res_cat = (proj.groupby(['Project Resource Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
proj_grade_cat = (proj.groupby(['Project Grade Level Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(proj_cat, 'Project Subject Category Tree')
plot_trend(proj_sub_cat, 'Project Subject Subcategory Tree')
plot_trend(proj_res_cat, 'Project Resource Category')
donor_master = donation_master.copy()

# One-hot encode Project Subject Category Tree
cat = (pd.DataFrame(donor_master['Project Subject Category Tree']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Literacy & Language', 'Math & Science', 'Applied Learning', 'Music & The Arts']
cat[~cat['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Subject Category Tree', 1)
                .join(pd.get_dummies(cat, prefix='Project Subject Category Tree', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Subject Category Tree_Other'] > 1, 'Project Subject Category Tree_Other'] = 1


# One-hot encode Project Subject Subcategory Tree
sub_cat = (pd.DataFrame(donor_master['Project Subject Subcategory Tree']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Literacy', 'Mathematics', 'Literature & Writing', 'Special Needs', 'Early Development',
            'Environmental Science']
sub_cat[~sub_cat['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Subject Subcategory Tree', 1)
                .join(pd.get_dummies(sub_cat, prefix='Project Subject Subcategory Tree', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Subject Subcategory Tree_Other'] > 1, 'Project Subject Subcategory Tree_Other'] = 1

# One-hot encode Project Resource Category
resrc = (pd.DataFrame(donor_master['Project Resource Category']
                     .str.split(', ').tolist())
        .stack().str.strip()
        .to_frame('col')
       )
keep_cols = ['Supplies', 'Technology', 'Books', 'Computers & Tablets']
resrc[~resrc['col'].isin(keep_cols)] = 'Other'

donor_master = (donor_master.drop('Project Resource Category', 1)
                .join(pd.get_dummies(resrc, prefix='Project Resource Category', prefix_sep='_')
                      .sum(level=0))
               )
donor_master.ix[donor_master['Project Resource Category_Other'] > 1, 'Project Resource Category_Other'] = 1
donor_master.head(10)
donor_master['In State'] = donor_master['School State'] == donor_master['Donor State']
donor_master['In State'] = donor_master['In State'].astype(int)
donor_master.drop(['School State', 'Donor State'], axis=1, inplace=True)
donor_master.head()
donor_master['School Percentage Free Lunch'] = donor_master['School Percentage Free Lunch'] / 100
donor_master.head()
donor_master['Donor Is Teacher'] = donor_master['Donor Is Teacher'].map(dict(Yes=1, No=0))
donor_master.head()
custom_bucket = [0, 179, 299, 999, 2500, 100000]
#custom_bucket = [0, 199, 399, 999, 2999, 100000]
custom_bucket_label = ['Vey Low', 'Low', 'Medium', 'High', 'Very High']
donor_master['Project Cost'] = pd.cut(donor_master['Project Cost'], custom_bucket, 
                                      labels=custom_bucket_label)

(donor_master['Project Cost'].value_counts()
                             .sort_index()
                             .iplot(kind='bar', xTitle = 'Project Cost', yTitle = "Project Count", 
                                    title = 'Distribution on Project Cost', color='green')
)

custom_bucket = [0, 4.99, 19.99, 99.99, 499.99, 1000000]
#custom_bucket = [0, 5, 25, 100, 300, 1000000]
custom_bucket_label = ['Vey Low', 'Low', 'Medium', 'High', 'Very High'] # Creating a dummy hierarchy
donor_master['Donation Amount'] = pd.cut(donor_master['Donation Amount'], custom_bucket, labels=custom_bucket_label)

(donor_master['Donation Amount'].value_counts()
                             .sort_index()
                             .iplot(kind='bar', xTitle = 'Donation Amount', yTitle = 'Donation Count',
                             title = 'Simulated Distribution on Donation Amount')
)
cat_cols = ['Project Grade Level Category', 'School Metro Type', 'Project Cost', 'Donation Amount']
donor_master = pd.get_dummies(data=donor_master, columns=cat_cols)
donor_master.head()
donor_master_final = donor_master.drop('Project ID', axis=1).copy()
all_cols = list(donor_master_final.columns)
all_cols.remove('Donor ID')
action = {col : 'max' for col in all_cols}
action['School Percentage Free Lunch'] = 'median'
donor_master_final = donor_master_final.groupby('Donor ID').agg(action).reset_index()
donor_master_final.set_index('Donor ID', inplace=True)
donor_master_final.fillna(0, inplace=True)
donor_master_final.head(10)
from sklearn.decomposition import PCA
pca = PCA()
projected = pca.fit(donor_master_final)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(0.9).fit(donor_master_final)
pca.n_components_
# temp = PCA(pca.n_components_).fit(donor_master_final)
# donor_master_final = temp.transform(donor_master_final)
%%time
def k_donor_segment(donors):
    kmeans = KMeans(init='k-means++', n_clusters=n_donor_segments, 
                    n_init=10, precompute_distances=True)
    kmeans.fit(donors)
    print('k-means intertia - {}'.format(kmeans.inertia_))

    lab = kmeans.labels_
    segments = Counter()
    segments.update(lab)

    return lab, segments

def g_donor_segment(donors):
    # Predict GMM clusters
    gm = GaussianMixture(n_components=n_donor_segments)
    gm.fit(donors)
    
    lab = gm.predict(donors)
    segments = Counter()
    segments.update(lab)

    return lab, segments

def d_donor_segment(donors):
    db = DBSCAN(eps=3, min_samples=10)
    db.fit(donors)

    lab = db.labels_
    segments = Counter()
    segments.update(lab)

    return lab, segments

label, donor_segment = k_donor_segment(donor_master_final)

donor_segment_mapping = {donor_id : donor_seg for donor_id, donor_seg 
                   in zip(list(donor_master_final.index), list(label))}

display(donor_segment.most_common(10))
col = ['Donation Amount_Vey Low', 'Donation Amount_Low', 'Donation Amount_Medium', 'Donation Amount_High',
       'Donation Amount_Very High', 'In State', 'Donor Is Teacher']
proj_master = donor_master.drop(col, axis=1).copy()

proj_donor_map = (proj_master.groupby('Project ID')['Donor ID']
                  .apply(list)
                  .to_frame('Donors')
                  .reset_index()
                 )
proj_donor_map['Donors'] = proj_donor_map['Donors'].apply(lambda x: list(set(x)))
proj_master = proj_master.merge(proj_donor_map, on='Project ID', how='inner')
proj_master.drop('Donor ID', axis=1, inplace=True)
proj_master.drop_duplicates('Project ID', keep='first')

proj_master.set_index('Project ID', inplace=True)
proj_master.fillna(0, inplace=True)
features, lables = proj_master.drop('Donors', axis=1), proj_master['Donors']


# Split the train and test dataset 
train_features, test_features, train_lables, test_lables = train_test_split(features, lables,
                                                                           test_size=.3)
# Split the test dataset into validatin and test dataset
test_features, valid_features, test_lables, valid_lables = train_test_split(test_features, test_lables,
                                                                           test_size=.5)

def one_hot_encode_labels(proj_donors, donor_segment_mapping):
    n_donor_segments = len(set(donor_segment_mapping.values()))
    lables = np.zeros(shape=(proj_donors.shape[0], n_donor_segments))
    
    def get_max(x):
        ''' For given row convert highest value to 1 and rest to zero
            For give
        '''
        max = np.unravel_index(x.argmax(), x.shape)
        x = x * 0
        x[max] = 1
        return x

    for i, values in enumerate(proj_donors):
        for val in values:
            segment = donor_segment_mapping[val]
            lables[i][segment] += 1
        lables[i] = get_max(lables[i])
        
    return lables

train_lables = one_hot_encode_labels(train_lables, donor_segment_mapping)
valid_lables = one_hot_encode_labels(valid_lables, donor_segment_mapping)
test_lables = one_hot_encode_labels(test_lables, donor_segment_mapping)
def build_nn(X_train, Y_train, X_valid, Y_valid,
             epochs=2000, batch_size=200,
             activation='relu',
             layer_size=[100, 50, 30],
             dropout=[.3, .2, 0]):
    
    n_input = X_train.shape[1]
    n_classes = Y_train.shape[1]
    init = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    
    # opt =  optimizers.Adam()
    # opt = optimizers.Adam(lr=0.0005, amsgrad=False)
    # opt = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # opt = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    opt = optimizers.Nadam(lr=0.0035, beta_1=0.1, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    checkpointer = ModelCheckpoint(filepath='donors.hdf5', verbose=False, save_best_only=True)
    model = Sequential() 
    for i, val in enumerate(range(len(layer_size))):
        if i == 0:
            model.add(Dense(layer_size[i], activation=activation, 
                            input_shape=(n_input,), kernel_initializer=init))
        else:
            model.add(Dense(layer_size[i], activation=activation, kernel_initializer=init))
        model.add(Dropout(dropout[i]))    

    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    logs = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                     validation_data=(X_valid, Y_valid), shuffle=True, verbose=0,
                    callbacks=[checkpointer])
    
    model.load_weights('donors.hdf5')
    score = model.evaluate(X_train, Y_train)
    print("Training Accuracy:", score[1])
    score = model.evaluate(X_valid, Y_valid)
    print("Validation Accuracy:", score[1])

    return model, logs
    
%%time
model, logs = build_nn(train_features.values, train_lables,
                       valid_features.values, valid_lables, 
                       epochs=300)

print ('Model Training Completed')

# Analyze accuracy over epochs
plt.plot(logs.history['acc'])
plt.plot(logs.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('epoch #')
plt.legend(['train', 'test'], loc='upper left')
plt.title('Trend of Accuracy')
plt.show()

# Analyze loss over epochs
plt.plot(logs.history['loss'])
plt.plot(logs.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('epoch #')
plt.legend(['train', 'validation'], loc='upper left')
plt.title('Trend of Loss')
plt.show()

score = model.evaluate(test_features.values, test_lables)
print("Testing Accuracy:", score[1])

# Predict Donor segment for each project
predict_segment = model.predict_classes(test_features.values)

# Generate an dictionary with a mapping of projects and potential donors
# Reverse engineer a mapping of donors and segments
donor_cluster = {value : [] for value in set(donor_segment_mapping.values())} 
{donor_cluster[val].append(key)for key, val in donor_segment_mapping.items()}

# Predicted donors
predicted_donors = {proj : donor_cluster[seg] for 
                    proj, seg in zip(list(test_features.index), list(predict_segment))}

print ('Prediction Completed')
# Let's display number of paired donors for some projects
sample = 10
for i, key in enumerate(predicted_donors):
    print('There are {:,} potential donors for {} project'
          .format(len(predicted_donors[key]), key))
    if i == sample: break
    