import threading
import time


class AsyncWrite(threading.Thread):
    def __init__(self, text, out):
        threading.Thread.__init__(self)
        self.text = text
        self.out = out

    def run(self):
        self.get_ready()
        f = open(self.out, "a")
        f.write(self.text + '\n')
        f.close()
        time.sleep(2)
        print("Finished Background file write to " + self.out)
        self.clean_up()

    def get_ready(self):
        pass

    def clean_up(self):
        pass

message = input("Enter a string to store:")

background = AsyncWrite(message, 'out.txt')
background.start()

background2 = AsyncWrite(message, 'out2.txt')
background2.start()

print("The program can continue while it writes in another thread")
print("100 + 400 = ", 100 + 400)

background.join()
print("Waited until thread was complete")

