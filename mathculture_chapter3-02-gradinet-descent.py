def gradient(x):

    return x ** 2 - 2
def gradient_descent(x=1, step=0.1, loop=100):

    for i in range(loop):

        x = x - step * gradient(x)

        if i % 10 == 0:

            print(i, "th loop x is ", x)
gradient_descent()
gradient_descent(100)
gradient_descent(10)
def newton_grad(x):

    return (x ** 2 - 2)/ (2 * x)
def newton(x=1, loop=100):

    for i in range(loop):

        x = x - newton_grad(x)

        if i % 10 == 0:

            print(i, "th loop x is ", x)
newton()
newton(100)
newton(0)
newton(-10)