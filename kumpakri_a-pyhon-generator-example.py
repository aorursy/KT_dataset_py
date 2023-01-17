def sub_generator(start):

    """

    Generates one sample

    """

    i = start

    while True:

        yield i

        i += 1
def batch_generator(sub_generator, batch_size):

    """

    Generates a batch of samples using the sub_generator

    """

    while True:

        batch = []

        for i in range(batch_size):

            batch.append(next(sub_generator))

        yield batch
# initialize the generators, nothing happens

batch_gen = batch_generator(sub_generator(start = 10), batch_size = 5)

# run the generator, code executes until the yield instruction

# generator remembers its state

print(next(batch_gen))

print("--------")

# generator starts where it ended last call

# runs until the yield instruction

print(next(batch_gen))