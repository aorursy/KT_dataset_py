import numpy as np
import pandas as pd
meta_data = pd.read_csv('../input/training_set_metadata.csv')
test_meta_data = pd.read_csv('../input/test_set_metadata.csv')
classes = np.unique(meta_data['target'])
classes_all = np.hstack([classes, [99]])

# create a dictionary {class : index} to map class number with the index 
# (index will be used for submission columns like 0, 1, 2 ... 14)
target_map = {j:i for i, j in enumerate(classes_all)}

# create 'target_id' column to map with 'target' classes
target_ids = [target_map[i] for i in meta_data['target']]
meta_data['target_id'] = target_ids
# Build probability arrays for both the galactic and extragalactic groups
galactic_cut = meta_data['hostgal_specz'] == 0
galactic_data = meta_data[galactic_cut]
extragalactic_data = meta_data[~galactic_cut]

galactic_classes = np.unique(galactic_data['target_id'])
extragalactic_classes = np.unique(extragalactic_data['target_id'])

# add class_99 (index = 14)
galactic_classes = np.append(galactic_classes, 14)
extragalactic_classes = np.append(extragalactic_classes, 14)
# Weighted probabilities for Milky Way galaxy
galactic_probabilities = np.zeros(15)
for x in galactic_classes:
    if(x == 14):
        galactic_probabilities[x] = 0.014845745
        continue
    if(x == 5):
        galactic_probabilities[x] = 0.196867058
        continue
    galactic_probabilities[x] = 0.197071799


# Weighted probabilities for Extra Galaxies
extragalactic_probabilities = np.zeros(15)
for x in extragalactic_classes:
    if(x == 14):
        extragalactic_probabilities[x] = 0.148880461
        continue
    if(x == 7):
        extragalactic_probabilities[x] = 0.155069005
        continue
    if(x == 1):
        extragalactic_probabilities[x] = 0.154666479
        continue
    extragalactic_probabilities[x] = 0.077340579
# Apply this prediction to test_meta_data table
import tqdm
def do_prediction(table):
    probs = []
    for index, row in tqdm.tqdm(table.iterrows(), total=len(table)):
        if row['hostgal_photoz'] == 0:
            prob = galactic_probabilities
        else:
            prob = extragalactic_probabilities
        probs.append(prob)
    return np.array(probs)

test_pred = do_prediction(test_meta_data)
test_df = pd.DataFrame(index=test_meta_data['object_id'], data=test_pred, columns=['class_%d' % i for i in classes_all])
test_df.to_csv('./naive_benchmark_weighted.csv')