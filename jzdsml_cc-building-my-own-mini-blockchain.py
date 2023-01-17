import datetime

from hashlib import sha256



#changed data to transactions



class Block:

    def __init__(self, transactions, previous_hash):

        self.time_stamp = datetime.datetime.now()

        self.transactions = transactions

        self.previous_hash = previous_hash

        self.nonce = 0

        self.hash = self.generate_hash()



    def generate_hash(self):

        block_header = str(self.time_stamp) + str(self.transactions) +str(self.previous_hash) + str(self.nonce)

        block_hash = sha256(block_header.encode())

        return block_hash.hexdigest()



    def print_contents(self):

        print("timestamp:", self.time_stamp)

        print("transactions:", self.transactions)

        print("current hash:", self.generate_hash())

        print("previous hash:", self.previous_hash) 
class Blockchain:

    def __init__(self):

        self.chain = []

        self.unconfirmed_transactions = []

        self.genesis_block()



    def genesis_block(self):

        transactions = []

        genesis_block = Block(transactions, "0")

        genesis_block.generate_hash()

        self.chain.append(genesis_block)



    def add_block(self, transactions):

        previous_hash = (self.chain[len(self.chain)-1]).hash

        new_block = Block(transactions, previous_hash)

        new_block.generate_hash()

        # proof = proof_of_work(block)

        self.chain.append(new_block)



    def print_blocks(self):

        for i in range(len(self.chain)):

            current_block = self.chain[i]

            print("Block {} {}".format(i, current_block))

            current_block.print_contents()



    def validate_chain(self):

        for i in range(1, len(self.chain)):

            current = self.chain[i]

            previous = self.chain[i-1]

            if(current.hash != current.generate_hash()):

                print("Current hash does not equal generated hash")

                return False

            if(current.previous_hash != previous.generate_hash()):

                print("Previous block's hash got changed")

                return False

        return True

 

    def proof_of_work(self, block, difficulty=2):

        proof = block.generate_hash()

        while proof[:2] != "0"*difficulty:

            block.nonce += 1

            proof = block.generate_hash()

        block.nonce = 0

        return proof

      
block_one_transactions = {"sender":"Alice", "receiver": "Bob", "amount":"50"}

block_two_transactions = {"sender": "Bob", "receiver":"Cole", "amount":"25"}

block_three_transactions = {"sender":"Alice", "receiver":"Cole", "amount":"35"}

fake_transactions = {"sender": "Bob", "receiver":"Cole, Alice", "amount":"25"}



local_blockchain = Blockchain()

local_blockchain.print_blocks()



local_blockchain.add_block(block_one_transactions)

local_blockchain.add_block(block_two_transactions)

local_blockchain.add_block(block_three_transactions)

local_blockchain.print_blocks()

local_blockchain.chain[2].transactions = fake_transactions

local_blockchain.validate_chain()
