# Imports and helper functions
import tensorflow as tf

def int_list_to_hex(l):
    return ''.join("{0:0{1}x}".format(x, 2) for x in l)

def int_list_to_string(l):
    return ''.join(chr(x) for x in l)
message_str = "Hello this is a secret message."
message = tf.constant([ord(c) for c in message_str], tf.uint8)

key_uint32 = tf.Variable(tf.random_uniform(message.shape, minval=0, maxval=2**8, dtype=tf.int32))
key = tf.cast(key_uint32, tf.uint8)

encrypt_xor = tf.bitwise.bitwise_xor(message, key)
decrypt_xor = tf.bitwise.bitwise_xor(encrypt_xor, key)

with tf.Session().as_default() as session:
    session.run(tf.global_variables_initializer())
    print('key:'.ljust(24), int_list_to_hex(key.eval()))
    print('message:'.ljust(24), int_list_to_string(message.eval()))

    ciphertext = encrypt_xor.eval()
    print('encrypted ciphertext:'.ljust(24), int_list_to_hex(ciphertext))

    plaintext = decrypt_xor.eval()
    print('decrypted plaintext:'.ljust(24), int_list_to_string(plaintext))
BLOCK_SIZE = 32
NUM_ROUNDS = 16

def feistel_network_encrypt_round(round_key, left_0, right_0):
    """Run one encryption round of a Feistel network.

    Args:
        round_key: The PRF is keyed with this round key.
        left_0: the left half of the input.
        right_0: the right half of the input.
    Returns:
        right n+1: the right half ouput.
        left n+1: the left half output.
    """
    # (Using bitwise inversion instead of a true PRF)
    f_ri_ki = tf.bitwise.invert(right_0)
    right_plusone = tf.bitwise.bitwise_xor(left_0, f_ri_ki)

    return right_0, right_plusone


def feistel_network_decrypt_round(round_key, left_plusone, right_plusone):
    """Run one decryption round of a Feistel network.

    Args:
        round_key: The PRF is keyed with this round key.
        left_plusone: the preceding left half of the input.
        right_plusone: the precedingright half of the input.
    Returns:
        left n-1: the decrypted left half.
        right n-1: the decrypted right half.
    """
    # (Using bitwise inversion instead of a true PRF)
    f_lip1_ki = tf.bitwise.invert(left_plusone)
    right_0 = tf.bitwise.bitwise_xor(right_plusone, f_lip1_ki)

    return right_0, right_plusone

def pkcs7_pad(text):
    # Not true PKCS #7 padding, only for demo purposes.
    val = BLOCK_SIZE - (len(text) % BLOCK_SIZE)
    return text + ('%d' % val) * val

def pkcs7_unpad(text):
    val = text[-1]
    return text[:(len(text) - int(text[-1]))]

message_str = pkcs7_pad("Hello this is a secret message.")
input_tensor = tf.constant([ord(c) for c in message_str], tf.uint8)

key_uint32 = tf.Variable(tf.random_uniform((NUM_ROUNDS,), minval=0, maxval=2**8, dtype=tf.int32))
key = tf.cast(key_uint32, tf.uint8)

with tf.Session().as_default() as session:
    session.run(tf.global_variables_initializer())

    # Keys here are used to seed the random shuffle.
    # Key is 16 bytes, one byte per round.
    # (Note: this does not follow the DES key scheduling algorithm).
    print('key:'.ljust(24), int_list_to_hex(key.eval()))
    print('padded message:'.ljust(24), int_list_to_string(input_tensor.eval()))
    
    # Encryption: split the input in half and run the network for 16 rounds.
    left, right = tf.split(input_tensor, num_or_size_splits=2)
    
    for round_num in range(NUM_ROUNDS):
        right, left = feistel_network_encrypt_round(key[round_num], left, right)
    
    print('encrypted ciphertext:'.ljust(24), int_list_to_hex(left.eval()) + int_list_to_hex(right.eval()))

    # Decryption: run the network in reverse.
    for round_num in range(NUM_ROUNDS):
        left, right = feistel_network_decrypt_round(key[round_num], left, right)
    
    print('decrypted plaintext:'.ljust(24), pkcs7_unpad(int_list_to_string(left.eval()) + int_list_to_string(right.eval())))