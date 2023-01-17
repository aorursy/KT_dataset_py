print(Q)
1 + 'abc'
2 / 0
L = [1, 2, 3]

L[1000]
try:

    print("this gets executed first")

except:

    print("this gets executed only if there is an error")
try:

    print("let's try something:")

    x = 1 / 0 # ZeroDivisionError

except:

    print("something bad happened!")
def safe_divide(a, b):

    try:

        return a / b

    except:

        return 1E100
safe_divide(1, 2)
safe_divide(2, 0)
safe_divide (1, '2')
def safe_divide(a, b):

    try:

        return a / b

    except ZeroDivisionError:

        return 1E100
safe_divide(1, 0)
safe_divide(1, '2')
raise RuntimeError("my error message")
def fibonacci(N):

    L = []

    a, b = 0, 1

    while len(L) < N:

        a, b = b, a + b

        L.append(a)

    return L
def fibonacci(N):

    if N < 0:

        raise ValueError("N must be non-negative")

    L = []

    a, b = 0, 1

    while len(L) < N:

        a, b = b, a + b

        L.append(a)

    return L
fibonacci(10)
fibonacci(-10)
N = -10

try:

    print("trying this...")

    print(fibonacci(N))

except ValueError:

    print("Bad value: need to do something else")
try:

    x = 1 / 0

except ZeroDivisionError as err:

    print("Error class is:  ", type(err))

    print("Error message is:", err)
class MySpecialError(ValueError):

    pass



raise MySpecialError("here's the message")
try:

    print("do something")

    raise MySpecialError("[informative error message here]")

except MySpecialError:

    print("do something else")
try:

    print("try something here")

except:

    print("this happens only if it fails")

else:

    print("this happens only if it succeeds")

finally:

    print("this happens no matter what")