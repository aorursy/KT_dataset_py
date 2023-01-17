class PowerOfN:
    def __init__(self, n, max=0):
        self.n = n
        self.max = max
        self.power = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.power <= self.max:
            result = self.n ** self.power
            self.power += 1
            return result
        else:
            self.power = 0
            raise StopIteration
powerOf3 = iter(PowerOfN(3, 4))
print(next(powerOf3))
print(next(powerOf3))
print(next(powerOf3))
print(next(powerOf3))
print(next(powerOf3))
try:
    print(next(powerOf3))
except StopIteration:
    pass
print(next(powerOf3))
print(next(powerOf3))