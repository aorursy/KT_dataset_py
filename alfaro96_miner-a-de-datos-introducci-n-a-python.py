foo = 10

bar = 5



print(foo)

print(bar)



foo

bar
class Foo:

    """Foo class."""



    def __init__(self, *, foo):

        """Constructor."""

        self.foo = foo
my_foo = Foo(foo="foo")

my_foo.foo
def even(numbers):

    """Filter even numbers."""

    return [number for number in numbers if number % 2 == 0]
numbers = [number * 3 for number in range(1, 10)]

numbers
even_numbers = even(numbers)

even_numbers
# Third party

import numpy as np

import pandas as pd
seed = 707005

np.random.seed(seed)
arr_py_1d = np.array([0, 1, 2, 3])

arr_py_1d
arr_gen_1d = np.arange(4)

arr_gen_1d
arr_py_2d = np.array([[0, 1, 2, 3],

                      [4, 5, 6, 7],

                      [8, 9, 10, 11]])

arr_py_2d
arr_gen_2d = np.arange(12).reshape(3, 4)

arr_gen_2d
arr_py_2d[1]
arr_py_2d[0][2]
arr_py_2d[0, 2]
arr_zeros = np.zeros((3, 3))

arr_zeros
arr_ones = np.ones((3, 3))

arr_ones
arr_eye = np.eye(3, 3)

arr_eye
arr_py_2d.shape
arr_py_2d.ndim
arr_py_2d.dtype.name
arr_py_2d + 3
arr_py_2d - 3
arr_py_2d * 3
arr_py_2d / 3
arr_py_2d.T
arr_py_2d[:, 2]
arr_py_2d[0, :]
arr_py_2d[:2, :2]
n = 1000



unif = np.random.uniform(size=n)

norm = np.random.normal(size=n)



days = ["monday", "tuesday", "friday"]

days = np.random.choice(days, size=n)
data = {"uniform": unif, "normal": norm, "days": days}

df = pd.DataFrame(data)
df
df.head(5)
df.sample(5, random_state=seed)
df.describe(include="number")
df.describe(include="object")
df.loc[0:5]
df.loc[0:5, ["uniform", "normal"]]
df.loc[(df.normal > 0) & (df.days == "monday")].head(5)
np.mean(df)
np.std(df)
df.replace(to_replace="friday", value="saturday").sample(5, random_state=seed)
# TODO: Introduce the code here
# TODO: Introduce the code here
# TODO: Introduce the code here
assert odd_comprehension(numbers) == odd_lambda(numbers) == odd_loop(numbers)
%timeit -r 5 -n 10 odd_comprehension(numbers)
%timeit -r 5 -n 10 odd_lambda(numbers)
%timeit -r 5 -n 10 odd_loop(numbers)
# TODO: Introduce the code here
expected_arr = np.array([[0, 0, 5],

                         [0, 0, 0],

                         [0, 0, 0]])
np.testing.assert_array_equal(arr, expected_arr)
# TODO: Introduce the code here
expected_arr = np.array([[2, 2, 2, 2, 2],

                         [2, 2, 2, 2, 2],

                         [2, 2, 2, 2, 2],

                         [2, 2, 2, 2, 2]])
np.testing.assert_array_equal(arr, expected_arr)
# TODO: Introduce the code here
expected_arr = np.array([[3, 0, 0],

                         [0, 3, 0],

                         [0, 0, 3]])
np.testing.assert_array_equal(arr, expected_arr)
arr = np.arange(25).reshape(5, 5).T

arr
# TODO: Introduce the code here
expected_arr = np.array([[6, 11, 16],

                         [7, 12, 17],

                         [8, 13, 18]])
np.testing.assert_array_equal(arr, expected_arr)
# TODO: Introduce the code here
df2.sample(5, random_state=seed)
# TODO: Introduce the code here
arr = df2.loc[[42, 119, 122, 178, 616, 649, 710, 730, 804], "exponential"]
expected_arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
np.testing.assert_array_equal(arr, expected_arr)