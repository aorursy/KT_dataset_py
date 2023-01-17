import numpy as np

from keras.utils import to_categorical
def load_data_sets(path):
    test_data = []
    test_label = []
    train_data = []
    train_label = []
    with open(path + 'voc_test.txt', 'r') as test_file:
        lines = test_file.readlines()
        for line in lines:
            data = line.split(' ')
            test_data.append(data[0].strip())
            test_label.append(data[1].strip())

    with open(path + 'voc_train.txt', 'r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            data = line.split(' ')
            train_data.append(data[0].strip())
            train_label.append(data[1].strip())

    all_data = train_data.copy()
    all_data.extend(test_data)
    all_data = "".join(all_data)
    encrypt_vocab = list(set(all_data))
    encrypt_vocab_map = {}
    i = 0
    for vocab in encrypt_vocab:
        encrypt_vocab_map[vocab] = i
        i += 1

    encrypt_vocab_map['<unk>'] = i
    encrypt_vocab_map['<pad>'] = i + 1

    all_label = train_label.copy()
    all_label.extend(test_label)
    all_label = "".join(all_label)
    decrypt_vocab = list(set(all_label))
    decrypt_vocab_map = {}
    inv_decrypt_vocab_map = {}
    i = 0
    for vocab in decrypt_vocab:
        decrypt_vocab_map[vocab] = i
        inv_decrypt_vocab_map[i] = vocab
        i += 1

    decrypt_vocab_map['<unk>'] = i
    inv_decrypt_vocab_map[i] = '<unk>'
    decrypt_vocab_map['<pad>'] = i + 1
    inv_decrypt_vocab_map[i + 1] = '<pad>'

    return train_data, train_label, test_data, test_label, encrypt_vocab_map, decrypt_vocab_map, inv_decrypt_vocab_map

def preprocess_data(X, Y, encrypt_vocab, decrypt_vocab, Tx, Ty):
    X = np.array([string_to_int(i, Tx, encrypt_vocab) for i in X])
    Y = [string_to_int(t, Ty, decrypt_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(encrypt_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(decrypt_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh


def string_to_int(string, length, vocab):
    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    # print (rep)
    return rep
# max input length and output length
Tx = 45
Ty = 15
train_data, train_label, test_data, test_label, encrypt_vocab_map, decrypt_vocab_map, inv_decrypt_vocab_map = load_data_sets("../input/")
train_X, train_Y, train_Xoh, train_Yoh = preprocess_data(train_data, train_label, encrypt_vocab_map, decrypt_vocab_map, Tx, Ty)

print("X.shape:", train_X.shape)
print("Y.shape:", train_Y.shape)
print("Xoh.shape:", train_Xoh.shape)
print("Yoh.shape:", train_Yoh.shape)

index = 10
print("Source data:", train_X[index][0])
print("Target data:", train_Y[index][1])
print()
print("Source after preprocessing (indices):", train_X[index])
print("Target after preprocessing (indices):", train_Y[index])
print()
print("Source after preprocessing (one-hot):", train_Xoh[index])
print("Target after preprocessing (one-hot):", train_Yoh[index])
