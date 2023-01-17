def check_prime(number):

    for divisor in range(2, int(number ** 0.5) + 1):

        if number % divisor == 0:

            return False

    return True



class Primes:

    def __init__(self, max):

        # the maximum number of primes we want generated

        self.max = max

        # start with this number to check if it is a prime.

        self.number = 1

        # No of primes generated yet. We want to StopIteration when it reaches max

        self.primes_generated = 0

    def __iter__(self):

        return self

    def __next__(self):

        self.number += 1

        if self.primes_generated >= self.max:

            raise StopIteration

        elif check_prime(self.number):

            self.primes_generated+=1

            return self.number

        else:

            return self.__next__()
prime_generator =Primes(10)
for x in prime_generator:

    print(x)
def Primes(max):

    number = 1

    generated = 0

    while generated < max:

        number += 1

        if check_prime(number):

            generated+=1

            yield number
prime_generator = Primes(10)

for x in prime_generator:

    print(x)
primes = (i for i in range(1,10) if check_prime(i))


for x in primes:

    print(x)
def triplet(n): # Find all the Pythagorean triplets between 1 and n

    for a in range(n):

        for b in range(a):

            for c in range(b):

                if a*a == b*b + c*c:

                    yield(a, b, c)
triplet_generator = triplet(100)
for x in triplet_generator:

    print(x)
triplet_generator = ((a,b,c) for a in range(100) for b in range(a) for c in range(b) if a*a == b*b + c*c)
for x in triplet_generator:

    print(x)