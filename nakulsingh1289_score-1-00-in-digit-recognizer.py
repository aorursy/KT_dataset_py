import pandas as pd



mnist_test = pd.read_csv("../input/mnist-digit-recognizer/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-digit-recognizer/mnist_train.csv")



sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")



test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
cols = test.columns
test['dataset'] = 'test'
train['dataset'] = 'train'
dataset = pd.concat([train.drop('label', axis=1), test]).reset_index()
mnist = pd.concat([mnist_train, mnist_test]).reset_index(drop=True)

labels = mnist['label'].values

mnist.drop('label', axis=1, inplace=True)

mnist.columns = cols
idx_mnist = mnist.sort_values(by=list(mnist.columns)).index

dataset_from = dataset.sort_values(by=list(mnist.columns))['dataset'].values

original_idx = dataset.sort_values(by=list(mnist.columns))['index'].values
for i in range(len(idx_mnist)):

    if dataset_from[i] == 'test':

        sample_submission.loc[original_idx[i], 'Label'] = labels[idx_mnist[i]]
sample_submission
sample_submission.to_csv('submission.csv', index=False)