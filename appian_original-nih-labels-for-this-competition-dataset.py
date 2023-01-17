import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/nihcsv/nih.csv')
print(df.shape)
df.head(10)
# class2 is the label given in NIH dataset (and imageIndex is filename of the image)
import os
print(os.listdir('../input/nihcsv'))

test = pd.read_csv('../input/nihcsv/nih_for_test1_images.csv')
print(test.shape)
test.head(10)
df[df.class1 == 'Normal'].class2.value_counts().to_frame().head(10).plot.bar()
df[df.class1 == 'Lung Opacity'].class2.value_counts().to_frame().head(10).plot.bar()
df[df.class1 == 'No Lung Opacity / Not Normal'].class2.value_counts().to_frame().head(10).plot.bar()
for class2, count2 in df.class2.value_counts().items():

    if count2 < 100: # ignore small count
        continue

    print('\n----- %s -----' % class2)

    _df = df[df.class2 == class2].class1
    for class1, count1 in _df.value_counts().items():
        ratio = count1 / _df.count()
        print('%d (%.2f%%) %s' % (count1, ratio * 100, class1))