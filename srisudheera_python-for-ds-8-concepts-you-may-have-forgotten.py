x = [1,2,3,4]
out = []
for item in x:
    out.append(item**2)
print(out)
# vs.
x = [1,2,3,4]
out = [item**2 for item in x]
print(out)
# lambda arguments: expression
double = lambda x: x * 2
print(double(5))
# Map
seq = [1, 2, 3, 4, 5]
result = list(map(lambda var: var*2, seq))
print(result)
# Filter
seq = [1, 2, 3, 4, 5]
result = list(filter(lambda x: x > 2, seq))
print(result)
import numpy as np
# np.arange(start, stop, step)
# np.arange(3, 7, 2)
# array([3, 5])
# np.linspace(start, stop, num)
# np.linspace(2.0, 3.0, num=5)
# array([ 2.0,  2.25,  2.5,  2.75, 3.0])
# df.drop('Row A', axis=0)
# df.drop('Column A', axis=1)
# df.shape
# of Rows, # of Columns