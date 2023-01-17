import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

#### Count words function
def count_(column, split_char=' '):
    words = []
    for category_name in column:
        category_words = category_name.split(split_char)
        for word in category_words:
            words.append(word)
    return words

#### Basic plotting functions
def plot_histogram(name, x, y=None, histfunc=None): # Histogram
    data = [go.Histogram(histfunc=histfunc, y=y,x=x, name=name)]
    py.iplot(data)

def plot_bar(name, x, y): # Bar plot
    data = [go.Bar(x=x, y=y, name=name)]
    py.iplot(data)
data = pd.read_csv('../input/train.csv', sep=',')
X, y = data.loc[:, data.columns != 'price'], data['price']
print(X.shape)
print(y.shape)
data.head()
# Counting absent data
count_nan = len(X) - X.count()
# Counting absent data percentage per category
print('Absent percentage:')
for k,v in count_nan.items():
    print(str(k) + ' = ' + str(count_nan[k]/len(X)) + '%')
categories = [k for k,v in count_nan.items()]
percentages = [v for k,v in count_nan.items()]
# Plot absent data per category
plot_bar(x=categories, y=percentages, name='Absent Data %')
# Get words
words = count_(column=X['name'])
# Plot the histogram of a set of product 'name' samples
plot_histogram(histfunc='count', x=words[:10000], name='Product Name Words Histogram')
# Get words
words = count_(X['category_name'].fillna(' '), split_char='/')
# Plot the histogram of a set of product 'catergory_name' samples
plot_histogram(histfunc='count', x=words[:10000], name='Category Words Histogram')
# Plot the Histogram of a sample product 'category_name' category
plot_histogram(histfunc='count', x=X.loc[:10000, 'brand_name'], name='Brand Name Histogram')
# Get words
words = count_(X['item_description'].fillna(' '))
# Plot the histogram of a set of product 'catergory_name' samples
plot_histogram(histfunc='count', x=words[:10000], name='Description Words Histogram')
# Plot the histogram of a set of product 'item_condition_id' samples
plot_histogram(histfunc='count', x=X.loc[:, 'item_condition_id'], name='Item Condition Histogram')
# Plot the histogram of a set of product 'shipping' samples
plot_histogram(histfunc='count', x=X.loc[:, 'shipping'], name='Shipping Histogram')
