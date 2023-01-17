#How to Softmax
import numpy as np
preds = [2.0, 1.0, 0.1]
print("Original vector: " + str(preds))
exponents = [np.exp(i) for i in preds] # e^i for all i (values) in the predictions vector using list comprehension
print("Exponent vector: " + str(exponents))
print("Does this vector sum to one?")

if sum(exponents) == 1:
    print("Yes")
else:
    print("No")
sum_of_exps = sum(exponents) # use this to normalize the exponents vector by dividing each value by the sum

softmax = [j/sum_of_exps for j in exponents] # divide all j in the vector exponents by sum_of_exps value
print(softmax)
print("Does this vector sum to one?")

if sum(softmax) == 1:
    print("Yes")
else:
    print("No")