import logging

import threading

import time



def thread_function(name):

    logging.info("Thread %s: starting", name)

    time.sleep(2)

    logging.info("Thread %s: finishing", name)



format = "%(asctime)s: %(message)s"

logging.basicConfig(

    format=format, level=logging.INFO, datefmt="%H:%M:%S"

)



print("Main: before creating thread")

x = threading.Thread(target=thread_function, args=(1,))

print("Main: before running thread")

x.start()

print("Main: wait for the thread to finish")

x.join()

print("Main: all done")
x = threading.Thread(target=thread_function, args=(1,))

x.start()
import random 

SENTINEL = object()



class Pipeline:

    """

    Class to allow a single element pipeline between producer and consumer.

    """

    def __init__(self):

        self.message = 0

        self.producer_lock = threading.Lock()

        self.consumer_lock = threading.Lock()

        self.consumer_lock.acquire()



    def get_message(self, name):

        logging.debug("%s:about to acquire getlock", name)

        self.consumer_lock.acquire()

        logging.debug("%s:have getlock", name)

        message = self.message

        logging.debug("%s:about to release setlock", name)

        self.producer_lock.release()

        logging.debug("%s:setlock released", name)

        return message



    def set_message(self, message, name):

        logging.debug("%s:about to acquire setlock", name)

        self.producer_lock.acquire()

        logging.debug("%s:have setlock", name)

        self.message = message

        logging.debug("%s:about to release getlock", name)

        self.consumer_lock.release()

        logging.debug("%s:getlock released", name)



def producer(pipeline):

    """Pretend we're getting a message from the network."""

    for index in range(10):

        message = random.randint(1, 101)

        logging.info("Producer got message: %s", message)

        pipeline.set_message(message, "Producer")



    # Send a sentinel message to tell consumer we're done

    pipeline.set_message(SENTINEL, "Producer")



def consumer(pipeline):

    """Pretend we're saving a number in the database."""

    message = 0

    while message is not SENTINEL:

        message = pipeline.get_message("Consumer")

        if message is not SENTINEL:

            logging.info("Consumer storing message: %s", message)



format = "%(asctime)s: %(message)s"

logging.basicConfig(format=format, level=logging.INFO,

                    datefmt="%H:%M:%S")



pipeline = Pipeline()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

    executor.submit(producer, pipeline)

    executor.submit(consumer, pipeline)
import concurrent.futures

import logging

import queue

import random

import threading

import time



def producer(queue, event):

    """Pretend we're getting a number from the network."""

    while not event.is_set():

        message = random.randint(1, 101)

        logging.info("Producer got message: %s", message)

        queue.put(message)



    logging.info("Producer received event. Exiting")



def consumer(queue, event):

    """Pretend we're saving a number in the database."""

    while not event.is_set() or not queue.empty():

        message = queue.get()

        logging.info(

            "Consumer storing message: %s (size=%d)", message, queue.qsize()

        )



    logging.info("Consumer received event. Exiting")



format = "%(asctime)s: %(message)s"

logging.basicConfig(format=format, level=logging.INFO,

                    datefmt="%H:%M:%S")



pipeline = queue.Queue(maxsize=10)

event = threading.Event()

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

    executor.submit(producer, pipeline, event)

    executor.submit(consumer, pipeline, event)



    time.sleep(0.1)

    logging.info("Main: about to set event")

    event.set()