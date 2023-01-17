# Ok let's first load code from the other kernel
import dill
f = open("../input/add", "rb")
add = dill.load(f)
f = open("../input/multiply", "rb")
multiply = dill.load(f)
# Now use the code loaded from the other kernel
add(1,2,3,4,5,6)
multiply(2,2,2,2,2,2)
