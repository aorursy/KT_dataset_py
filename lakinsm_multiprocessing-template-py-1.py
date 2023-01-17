#!/usr/bin/env python3

# Author: Steven Lakin
# Kaggle Facebook V Challenge, May 2016

#############
## Imports ##
#############
import argparse
import multiprocessing as mp
import sys
import logging
import resource


##########
## Vars ##
##########
chunksize = 20000000  # limit memory consumption by reading in blocks
overall = 0  # counter for stderr writing


###############
## Functions ##
###############
def input_parse():
    """ Parses an input file.
    This script only accepts stdin, so use cat file | script.py for correct functionality.
    :return: generator yielding split observations
    """
    for x in range(chunksize):
        line = sys.stdin.readline()
        if not line:
            return  # stop iteration
        yield line.rstrip().split(',')


def worker(chunk):
    """ This code block is executed many times across the data block (each worker receives a chunk of that block).
    The predictions are written to the logging cache (this is because writing to stdout produces thrashing).
    The logging cache is then flushed on every iteration of the outer loop.
    :param chunk: a chunk of reads divided amongst the pool of parallel workers
    :return: void
    """
    global my_global_object
    for checkin in chunk:
        # Do something
        row_id = checkin[0]
        p1 = 'first_prediction'
        p2 = 'second_prediction'
        p3 = 'third_prediction'
        logging.info('{},{} {} {}'.format(row_id, p1, p2, p3))  # Output format


def split(a, n):
    """ Splits an input list into n equal chunks; this works even if modulo > 0.
    :param a: list of arbitrary length
    :param n: number of groups to split into
    :return: generator of chunks
    """
    k, m = int(len(a) / n), len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def current_mem_usage():
    """
    :return: current memory usage in MB
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024



##############
## ArgParse ##
##############
parser = argparse.ArgumentParser('cat test_file.csv | multiprocessing_template.py')
parser.add_argument('-n', '--num_process', type=int, default=1, help='Number of processes to run in parallel')
parser.add_argument('-s', '--skip_lines', type=int, default=0, help='Number of header lines to skip in input file')


##########
## Main ##
##########
if __name__ == '__main__':
    mp.freeze_support()
    ## Parse the arguments using ArgParse
    args = parser.parse_args()

    ## Input must be on stdin; raise error if this is not the case
    if sys.stdin.isatty():
        raise IOError('Input must be on stdin.  Use stream redirect for correct functionality: cat file | script.py')
    else:
        for skip in range(args.skip_lines):
            _ = sys.stdin.readline()  # Throw out header lines

    ## Setup the logger for output of predictions to stdout.  This is necessary because writing directly to stdout
    ## in parallel causes thrashing and variable results.  The logger caches observations passed to it on every loop
    ## and flushes to stdout after the observations have been processed.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)

    logging.info('row_id,place_id')

    ## Read in each file chunk, process, and output predictions.  Chunk size should be set such that
    ## the block size doesn't overflow memory.
    pool = mp.Pool(processes=args.num_process)  # create pool of workers for parallel processing
    while True:
        chunks = [z for z in split([x for x in input_parse()], args.num_process)]  # divide reads into chunks
        sys.stderr.write('\nMemory used: {}MB'.format(current_mem_usage()))
        check = sum([len(x) for x in chunks])  # this is the break condition for the while loop (count of lines)
        overall += check  # add to overall read count for reporting to stderr
        sys.stderr.write('\nTotal observations processed {}, screening...'.format(overall))
        if check is 0:
            pool.close()
            pool.join()
            pool.terminate()
            del pool
            break
        res = pool.map(worker, chunks)  # All workers must finish before proceeding.
        handler.flush()  # flush the logging cache to stdout

        del chunks  # remove chunks from memory.  Otherwise memory usage will be doubled.
        if check < chunksize:
            pool.close()  # ask nicely
            pool.join()  # sigterm
            pool.terminate()  # sigkill
            del pool  # make sure pool is cleared
            break
        del check
        del res
        sys.stderr.write('\nFinished block.  Loading next chunk...\n')
    sys.stderr.write('\nTotal observations processed {}'.format(overall))