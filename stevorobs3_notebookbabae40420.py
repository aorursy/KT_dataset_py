# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', sep='\t', low_memory=False)

df.head(10)

# Any results you write to the current directory are saved as output.
df.dtypes
cleaned_df=df[df['origins'].notnull() & df['product_name'].notnull()]

milk_products=cleaned_df.loc[cleaned_df['product_name'].str.contains('milk')]



milk_products[milk_products['purchase_places'].notnull()]['purchase_places']





df.head