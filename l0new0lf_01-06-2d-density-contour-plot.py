import seaborn as sns

iris = sns.load_dataset("iris")



sns.jointplot(x="petal_length", y="petal_width", data=iris, kind="kde");