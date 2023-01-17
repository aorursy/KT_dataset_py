import pandas as pd

food = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')
food.head(5)
food.shape[0]
food.shape[1]
food.columns
food.columns[105]
food.dtypes[105]
food.index
food["product_name"][105]