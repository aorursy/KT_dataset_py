# We will only need OS and Pandas for this one

import os

import pandas as pd



# Path names

BASE_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"

TRAIN_PATH = BASE_PATH + 'stage_2_train.csv'

TEST_PATH = BASE_PATH + 'stage_2_sample_submission.csv'



# All labels that we have to predict in this competition

targets = ['epidural', 'intraparenchymal', 

           'intraventricular', 'subarachnoid', 

           'subdural', 'any']
# File sizes and specifications

print('\n# Files and file sizes')

for file in os.listdir(BASE_PATH)[2:]:

    print('{}| {} MB'.format(file.ljust(30), 

                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))
train_df = pd.read_csv(TRAIN_PATH)

train_df['ImageID'] = train_df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.png'

label_lists = train_df.groupby('ImageID')['Label'].apply(list)
train_df[train_df['ImageID'] == 'ID_0002081b6.png']
def prepare_df(path, train=False, nrows=None):

    """

    Prepare Pandas DataFrame for fitting neural network models

    Returns a Dataframe with two columns

    ImageID and Labels (list of all labels for an image)

    """ 

    df = pd.read_csv(path, nrows=nrows)

    

    # Get ImageID and type for pivoting

    df['ImageID'] = df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.png'

    df['type'] = df['ID'].str.split("_", n = 3, expand = True)[2]

    # Create new DataFrame by pivoting

    new_df = df[['Label', 'ImageID', 'type']].drop_duplicates().pivot(index='ImageID', 

                                                                      columns='type', 

                                                                      values='Label').reset_index()

    return new_df
# Convert dataframes to preprocessed format

train_df = prepare_df(TRAIN_PATH, train=True)

test_df = prepare_df(TEST_PATH)
print('Training data: ')

display(train_df.head())



print('Test data: ')

test_df.head()
# Save to CSV

train_df.to_csv('clean_train_df.csv', index=False)

test_df.to_csv('clean_test_df.csv', index=False)
def create_submission_file(IDs, preds):

    """

    Creates a submission file for Kaggle when given image ID's and predictions

    

    IDs: A list of all image IDs (Extensions will be cut off)

    preds: A list of lists containing all predictions for each image

    

    Returns a DataFrame that has the correct format for this competition

    """

    sub_dict = {'ID': [], 'Label': []}

    # Create a row for each ID / Label combination

    for i, ID in enumerate(IDs):

        ID = ID.split('.')[0] # Remove extension such as .png

        sub_dict['ID'].extend([f"{ID}_{target}" for target in targets])

        sub_dict['Label'].extend(preds[i])

    return pd.DataFrame(sub_dict)
# Finalize submission files

train_sub_df = create_submission_file(train_df['ImageID'], train_df[targets].values)

test_sub_df = create_submission_file(test_df['ImageID'], test_df[targets].values)
print('Back to the original submission format:')

train_sub_df.head(6)