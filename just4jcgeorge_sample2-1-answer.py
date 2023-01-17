height = [1.73, 1.68, 1.71, 1.89, 1.79]
height
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
weight
#TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'

weight / height ** 2
import numpy as np
np_height = np.array(height)
np_height
np_weight = np.array(weight)
np_weight
bmi = np_weight / np_height ** 2
bmi
height = [1.73, 1.68, 1.71, 1.89, 1.79]
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
#TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'

weight / height ** 2
np_height = np.array(height)
np_weight = np.array(weight)
np_weight / np_height ** 2
np.array([1.0, "is", True])
python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])
python_list + python_list
numpy_array + numpy_array
bmi
bmi[1]
bmi > 23
bmi [bmi > 23]