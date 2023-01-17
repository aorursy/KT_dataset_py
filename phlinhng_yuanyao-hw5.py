def createCounter():

    def counter():

        counter.calls += 1

        return counter.calls

    counter.calls = 0

    return counter
counterA = createCounter()

print(counterA(), counterA(), counterA(), counterA(), counterA()) # 1 2 3 4 5

counterB = createCounter()

if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:

    print('Pass!')

else:

    print('Fail!')