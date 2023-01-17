x = -15



if x == 0:

    print(x, "is zero")

elif x > 0:

    print(x, "is positive")

elif x < 0:

    print(x, "is negative")

else:

    print(x, "is unlike anything I've ever seen...")
for N in [2, 3, 5, 7]:

    print(N, end=' ') # print all on same line
for i in range(10):

    print(i, end=' ')
# range from 5 to 10

list(range(5, 10))
# range from 0 to 10 by 2

list(range(0, 10, 2))
i = 0

while i < 10:

    print(i, end=' ')

    i += 1
for n in range(20):

    # if the remainder of n / 2 is 0, skip the rest of the loop

    if n % 2 == 0:

        continue

    print(n, end=' ')
a, b = 0, 1

amax = 100

L = []



while True:

    (a, b) = (b, a + b)

    if a > amax:

        break

    L.append(a)



print(L)
L = []

nmax = 30



for n in range(2, nmax):

    for factor in L:

        if n % factor == 0:

            break

    else: # no break

        L.append(n)

print(L)