from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

aws_id = user_secrets.get_secret("aws_access_key_id")

aws_key = user_secrets.get_secret("aws_secret_access_key")

aws_region = user_secrets.get_secret("aws_region")



#If the Secrets Add-on works properly, we will see the current region here

# By the way, we do not need a region, because S3 allows us to create Global buckets too



aws_region
import boto3

import uuid

import os



s3 = boto3.resource(

    's3',

    aws_access_key_id=aws_id,

    aws_secret_access_key=aws_key,

)



s3_client = boto3.client(

    's3',

    aws_access_key_id=aws_id,

    aws_secret_access_key=aws_key,

)

bucket_name = ''.join(['kaggle2020', str(uuid.uuid4())])

bucket_response = s3_client.create_bucket(Bucket=bucket_name)



response = s3_client.list_buckets()



# Output the bucket names

#print('Existing buckets:')

#for bucket in response['Buckets']:

#    print(f'  {bucket["Name"]}')





for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

#        print(os.path.join(dirname, filename))

        file_name = os.path.join(dirname, filename)

        response = s3_client.upload_file(file_name, bucket_name, filename)



# Let us get some feedback: the list of the objects of our Bucket is very verbose and we can

# check that everything is OK



s3_client.list_objects_v2(Bucket=bucket_name)
# Finally, we will delete the bucket to let everything as it was before



bucket = s3.Bucket(bucket_name)

# suggested by Jordon Philips 

bucket.objects.all().delete()

s3_client.delete_bucket(Bucket=bucket_name)