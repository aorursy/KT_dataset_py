#Given a probability of each character, return a likely character, one-hot encoded

def sample(prediction):

    r = random.uniform(0,1)

    s = 0

    char_id = len(prediction) - 1

    for i in range(len(prediction)):

        s += prediction[i]

        if s >= r:

            char_id = i

            break

    char_one_hot = np.zeros(shape=[char_size])

    char_one_hot[char_id] = 1.0

    return char_one_hot