[i for i in range(20) if i % 3 > 0]
L = []

for n in range(12):

    L.append(n ** 2)

L
[n ** 2 for n in range(12)]
[(i, j) for i in range(2) for j in range(3)]
[val for val in range(20) if val % 3 > 0]
L = []

for val in range(20):

    if val % 3:

        L.append(val)

L
val = -10

val if val >= 0 else -val
[val if val % 2 else -val

 for val in range(20) if val % 3]
{n**2 for n in range(12)}
{a % 3 for a in range(1000)}
{n:n**2 for n in range(6)}
(n**2 for n in range(12))