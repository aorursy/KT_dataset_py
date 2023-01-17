print("Hello Kaggle!")
from matplotlib import pyplot as plt
%matplotlib inline

plt.plot(list(range(-5,6)),[i**3 for i in range(-5,6)],color='purple')
plt.xlabel('x')
plt.ylabel('y=x^3')
plt.title('A portion of the graph of y=x^3')
plt.show()
print("Yay! It works! You rock, Kaggle!")