# ASCII - Plain Ordinary English Text
# American Standard Code for Information Interchange


name = "ABC abc KEVIN GEEK!?!?!?"
for c in name:
    print(
        bin(ord(c)), # binary version of the ordinal number
        ord(c), # ordinal number in ascii for the letter/character
        c # THe letter itself
    )

class ASCII():

    def __init__(self):
        self.columns = 8
        self.limit = 128

    def print_matrix(self):
        print("\n".join( [
            self.title(),
            self.get_columns(),
            self.row_two(),
            self.body(), ]))

    def title(self):
        return "ASCII Matrix:"


    def get_columns(self):
        output = ["DECIMAL", ""]
        for column in range(0,self.columns):
            output.append(" {} ".format(column))
        return "\t".join(output)

    def row_two(self):
        output=["", "BINARY"]
        for column in range(0, self.columns):
            output.append("{0:03b}".format(column))
        return "\t".join(output)

    def body(self):
        output = []
        for high_nibble in range(0, self.limit // self.columns):
            output.append(self.row(high_nibble))
        return "\n".join(output)

    def row(self, high_nibble):
        r = high_nibble * self.columns
        output = []
        output.append(" {0:03d}".format(r))
        output.append("{0:04b}".format(high_nibble))

        for low_nibble in range(0, self.columns):
            output.append(self.column(r, low_nibble))
        return "\t".join(output)

    def column(self, r, low_nibble):
        v = r + low_nibble
        if v <= 27:
            return "^{}={}".format(chr(v+64),v)
        else:
            return "'{}'={}".format(chr(v),v)


if __name__ == "__main__":
    ASCII().print_matrix()

