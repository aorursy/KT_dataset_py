!head -2 train.csv > small_train.csv
import pandas as pd
small_train = pd.read_csv('../input/small-train/small_train.csv')

print(small_train)
types_dict = small_train.dtypes.to_dict()

types_dict
types_dict = {'id': 'int32',

             'item_nbr': 'int32',

             'store_nbr': 'int8',

             'unit_sales': 'float32'}
grocery_train = pd.read_csv('train.csv', low_memory=True, dtype=types_dict)
os.makedirs('tmp', exist_ok=True)  # Make a temp dir for storing the feather file

# Save feather file, requires pandas 0.20.0 at least:

grocery_train.to_feather('./tmp/grocery_train_raw')
grocery_train = pd.read_feather('./tmp/train_sub_raw')