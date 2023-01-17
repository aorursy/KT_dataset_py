import math

from random import randrange


def getR(bitLength):

    """Calculates the needed order r.

    

    Args:

        param bitLength (integer): The number of bits to encode.



    Returns:

        Order r as integer

    """

    i = 0

    while int(math.pow(2, i) - 1) - i < bitLength:

        i += 1



    return i



def hammingCode(number):

    """Higher order function which:

        1. Takes an integer as input

        2. Calculates the parity check matrix

        3. Calculates the generator matrix

        4. Encodes the number

        5. Decodes the number

    

    Args:

        param number (integer): The number to encode and decode.



    Returns:

        Nothing.

    """

    

    if number < 2:

        print("Please provide an integer greater or equal to two.")

        return

    

    # The number of bits needed to represend the parameter number as binary list.

    bitLength = number.bit_length()

    

    # calculate the needed order r.

    r = getR(bitLength)

    

    # the row length of the matrices.

    length = int(math.pow(2, r) - 1)

        

    # initialize the parity check matrix

    parityCheck = [[False for x in range(r)] for y in range(length)]

    

    # initialize the generator matrix

    generator = [[False for x in range(length)] for y in range(length - r)];

    

    # initialize the list of error codes

    errorCodes = [[False for x in range(length)] for y in range(length + 1)];

        

    def createParityCheckMatrix():

        """Calculates the parity check matrix

        

        """

        currentRow = 0

        currentIdentityRow = length - r

    

        for i in range(length, 0, -1):



            row = 0



            if i & (i -1) == 0:            

                row = currentIdentityRow

                currentIdentityRow += 1

            else:

                row = currentRow

                currentRow += 1



            for j in range(r):

                parityCheck[row][j] = (i & (1 << (r - 1 - j))) != 0

                

    

    def greateGeneratorMatrix():

        """Calculates the generator matrix

        

        """

        dataLength = length - r

        

        for i in range(dataLength):

            # Insert line of identity matrix.

            

            for j in range(dataLength):

                generator[i][j] = i == j

                

            for j in range(r):

                generator[i][dataLength + j] = parityCheck[i][j]

                

                

    def precalculateSyndromesAndErrorCodes():

        """Calculates the error codes

        

        """

        

        for i in range(length):

            code = [False for x in range(length)]

            

            code[i] = True

            

            syndrome = booleanArrayToInt(getSyndrome(code))

            errorCodes[syndrome][i] = True;

            

    def encode(data):

        """Encodes a given binary array.



        Args:

            param data (binary list): The list to encode



        Returns:

            The encoded list.

        """

        if len(data) != length - r:

            return []

        

        output = [False for x in range(length)]

        

        for i in range(len(data)):

            if data[i] == False:

                continue

                

            row = generator[i]

            

            for j in range(length):

                output[j] ^= row[j]

                

        return output

    



    def decode(codeword):

        """Decodes a given encoded binary array.



        Args:

            param codeword (binary list): The list to decode



        Returns:

            The decoded list.

        """

        

        if len(codeword) != length:

            return []

        

        syndrome = getSyndrome(codeword)

        syndromeInt = booleanArrayToInt(syndrome)

        errorCode = errorCodes[syndromeInt]

        

        corrected = [False for x in range(length)]

        

        for i in range(length):

            corrected[i] = codeword[i] ^ errorCode[i]

            

        return corrected[0:(length - r)]

    

    def getSyndrome(codeword):

        """Returns the error list for a given codeword



        Args:

            param codeword (binary list): The list to calculate the errors from



        Returns:

            The error list.

        """

        if len(codeword) != length:

            return []

        

        error = [False for x in range(r)]

        

        for i in range(r):

            for j in range(length):

                error[i] ^= codeword[j] & parityCheck[j][i]

                

        return error

    

    def booleanArrayToInt(booleans):

        """Converts a boolean list to an integer.



        Args:

            param booleans (binary list): The list to convert to an integer



        Returns:

            The calculated integer

        """

        n = 0

        

        for i in range(len(booleans)):

            n = (n << 1) | int(booleans[i] == True) 

            

        return n

    

    def intToBooleanArray():

        """Converts an integer to a list of booleans.

        The most significant bit is on the left hand side of the list.



        Returns:

            The calculated list of booleans.

        """

        binString = format(number, '04b').rjust(length - r, '0')

        return [x == '1' for x in binString]

        

    createParityCheckMatrix()

    greateGeneratorMatrix()

    precalculateSyndromesAndErrorCodes()

    

    bitArray = intToBooleanArray()

    print('1. The value to encode is: {}'.format(number))

    print('-----------------------------')    

    print('')

    

    print('2. The parity check matrix is:')

    print('------------------------------')

    print(parityCheck)

    print('')

    

    print('3. The generator matrix is:')    

    print('---------------------------')

    print(generator)

    print('')

    

    print('4. The number as binary list is:')  

    print('--------------------------------')

    print(bitArray)

    print('')

    

    encoded = encode(bitArray)

    print('5. The encoded number as binary list is:')      

    print('----------------------------------------')

    print(encoded)    

    print('')

    

    index = randrange(length - 1)

    print('6. Simulating Bit-Flip at index: {}'.format(index))      

    print('------------------------------------')   

    print('')

    

    encoded[index] =  not encoded[index]

    print('7. The encoded number as binary list is now:')      

    print('--------------------------------------------')

    print(encoded)    

    print('')

    

    decoded = decode(encoded)

    print('8. The decoded number as binary list is:')    

    print('----------------------------------------')

    print(decoded)   

    print('')

    

    print('9. The decoded value as number is: {}'.format(booleanArrayToInt(decoded)))   

    print('-------------------------------------')
hammingCode(15)