f = open('/kaggle/input/stanford-cars-for-learn/stanford_cars/0.1.0/label.labels.txt', 'r')

lines = [line[:-1] for line in f.readlines()]

f.close()
lines