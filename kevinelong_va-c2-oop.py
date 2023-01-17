class Counter:
        def __init__(self):
                self.count = 0

        def increment(self):
                self.count += 1

        def decrement(self):
                self.count -= 1

        def reset(self):
                 self.count = 0

# TEST – Create an instance of the Class
counter = Counter()
# the initial counter value is 0

counter.increment()
counter.increment()
counter.increment()
# the counter's value is now 3
print(counter.count)

counter.decrement()
# the counter's value is now 2
print(counter.count)

counter.reset()
# the counter's value is now 0
print(counter.count)
